import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from httpx import Response

import src.api.main as main
from src.api.schemas import CUSTOMER_ID_MAX_LENGTH, MAX_MONTHLY_CHARGES, MAX_TENURE_MONTHS
from src.data.config import PROJECT_ROOT
from tests.test_action_engine import CRITICAL_CUSTOMER, LOW_CUSTOMER
from tests.test_local_explainer import HIGH_RISK_TEST_INDEX, LOW_RISK_TEST_INDEX


@pytest.fixture(scope="module")
def client():
    """One real app startup (builds the SHAP+LIME context once, ~1.9s),
    reused by every test below. The `with` form is required to actually
    trigger `lifespan` -- a bare TestClient(app) skips startup entirely."""
    with TestClient(main.app) as c:
        yield c


def _customer_payload(client: TestClient, index: int, customer_id: str | None = None) -> dict:
    """Build a JSON-serializable request payload from a real test-split row.

    Mirrors tests/test_local_explainer.py's _row(context, index) pattern,
    plus numpy-scalar -> native coercion (TestClient.post(json=...) can't
    serialize numpy dtypes directly).
    """
    context = client.app.state.explainer_context
    row = context["X_test"].loc[[index]].iloc[0].to_dict()
    payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    if customer_id:
        payload["customerID"] = customer_id
    return payload


def _customer_payload_stub() -> dict:
    """A structurally valid CustomerPayload body -- used by tests where the
    exact customer doesn't matter (e.g. the 503 short-circuits before any of
    it is used)."""
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


def _missing_model_error() -> FileNotFoundError:
    """A realistic, path-bearing startup failure -- shared by every test
    that simulates a missing/unreadable model artifact, so the path-leak
    assertions in this module actually exercise a message that could leak
    a real filesystem path if main.py regressed."""
    missing_path = PROJECT_ROOT / "models" / "churn_model_metadata.json"
    return FileNotFoundError(f"[Errno 2] No such file or directory: '{missing_path}'")


def test_health_returns_ok_and_model_loaded_true(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "startup_error": None}


def test_health_reachable_before_lifespan_runs():
    """app.state is initialized in create_app() itself, not only inside
    `lifespan` -- so /health must stay a genuine "200 always" liveness check
    even hit before startup runs (a bare TestClient(app), no `with`, never
    triggers `lifespan` at all)."""
    bare_client = TestClient(main.create_app())  # deliberately no `with` -- lifespan never runs
    try:
        response = bare_client.get("/health")
    finally:
        bare_client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False, "startup_error": None}


def test_explain_matches_verified_high_risk_example(client):
    payload = _customer_payload(client, HIGH_RISK_TEST_INDEX, "5178-LMXOP")
    response = client.post("/explain", json=payload)
    assert response.status_code == 200

    body = response.json()
    shap_drivers = body["shap_top_drivers"]
    assert {d["feature"] for d in shap_drivers} == {"tenure", "Contract", "InternetService"}
    assert all(d["direction"] == "increases" for d in shap_drivers)

    # Full response-schema check (CLAUDE.md Sec 9: API tests confirm the
    # expected schema), not just feature names/directions.
    first_shap = shap_drivers[0]
    assert isinstance(first_shap["customer_value"], (str, int, float))
    assert isinstance(first_shap["shap_value"], float)
    assert isinstance(first_shap["reason"], str) and first_shap["reason"]

    first_lime = body["lime_top_drivers"][0]
    assert isinstance(first_lime["feature"], str) and first_lime["feature"]
    assert isinstance(first_lime["lime_weight"], float)
    assert isinstance(first_lime["reason"], str) and first_lime["reason"]
    assert body["customerID"] == "5178-LMXOP"


def test_explain_matches_verified_low_risk_example(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX, "4376-KFVRS")
    response = client.post("/explain", json=payload)
    assert response.status_code == 200

    body = response.json()
    shap_drivers = body["shap_top_drivers"]
    assert {d["feature"] for d in shap_drivers} == {"Contract", "tenure", "OnlineSecurity"}
    assert all(d["direction"] == "decreases" for d in shap_drivers)

    lime_features = [d["feature"] for d in body["lime_top_drivers"]]
    assert any("Contract" in f for f in lime_features)
    assert any("tenure" in f for f in lime_features)


def test_explain_missing_required_field_returns_422(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    del payload["Contract"]
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_unseen_category_returns_422(client):
    """Contract is Literal-typed (src/api/schemas.py) -- an unseen value is
    rejected by Pydantic itself, before explain_customer ever runs. The
    custom validation-error handler (src/api/main.py) strips the echoed
    rejected value from the response, so it must name the failing field
    without ever reflecting the submitted value back."""
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["Contract"] = "Lifetime"
    response = client.post("/explain", json=payload)
    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any("Contract" in err["loc"] for err in errors)
    assert all("input" not in err for err in errors)
    assert "Lifetime" not in response.text


def test_explain_large_unseen_value_is_not_echoed_back(client):
    """The same validation-error handler must not amplify a large rejected
    payload back into the response body -- the concrete scenario the
    handler exists to close."""
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["Contract"] = "x" * 100_000
    response = client.post("/explain", json=payload)
    assert response.status_code == 422
    assert len(response.text) < 10_000


def test_explain_rejects_unrecognized_field(client):
    """CustomerPayload's extra="forbid" turns a typo'd/unknown field into a
    self-documenting 422 rather than a silently-dropped value."""
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["NotARealColumn"] = "surprise"
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_rejects_tenure_above_upper_bound(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["tenure"] = MAX_TENURE_MONTHS + 1
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_rejects_infinite_monthly_charges(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["MonthlyCharges"] = "Infinity"
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_rejects_monthly_charges_above_upper_bound(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["MonthlyCharges"] = MAX_MONTHLY_CHARGES + 1
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_rejects_customer_id_above_max_length(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX, "x" * (CUSTOMER_ID_MAX_LENGTH + 1))
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_omitted_total_charges_defaults_to_zero(client, monkeypatch):
    """Verifies the actual normalization (CustomerPayload.to_customer_dict),
    not just that the request happens to return 200."""
    captured = {}

    def _capture(customer, context=None):
        captured.update(customer)
        return {"shap_top_drivers": [], "lime_top_drivers": []}

    monkeypatch.setattr(main, "explain_customer", _capture)

    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    del payload["TotalCharges"]
    response = client.post("/explain", json=payload)

    assert response.status_code == 200
    assert captured["TotalCharges"] == 0.0


def test_explain_accepts_raw_int_senior_citizen_encoding(client):
    """SeniorCitizen accepts both the raw CSV's native 0/1 int encoding and
    the cleaned-data "Yes"/"No" string form (src/api/schemas.py docstring)."""
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["SeniorCitizen"] = 0
    response = client.post("/explain", json=payload)
    assert response.status_code == 200


def test_explain_reuses_context_across_requests(client):
    context_before = client.app.state.explainer_context
    client.post("/explain", json=_customer_payload(client, HIGH_RISK_TEST_INDEX))
    client.post("/explain", json=_customer_payload(client, LOW_RISK_TEST_INDEX))
    assert client.app.state.explainer_context is context_before


def test_explain_concurrent_requests_do_not_error(client):
    """Integration-level smoke test: real SHAP+LIME explainer, several
    concurrent /explain calls, nothing raises. This does NOT by itself prove
    app.state.explain_lock is serializing anything -- see
    test_explain_lock_serializes_concurrent_calls for that."""
    payloads = [
        _customer_payload(client, HIGH_RISK_TEST_INDEX if i % 2 == 0 else LOW_RISK_TEST_INDEX)
        for i in range(6)
    ]

    def _post(payload: dict) -> Response:
        return client.post("/explain", json=payload)

    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(_post, payloads))

    assert all(r.status_code == 200 for r in responses)
    for i, response in enumerate(responses):
        expected = (
            {"tenure", "Contract", "InternetService"} if i % 2 == 0
            else {"Contract", "tenure", "OnlineSecurity"}
        )
        assert {d["feature"] for d in response.json()["shap_top_drivers"]} == expected


def test_explain_lock_serializes_concurrent_calls(client, monkeypatch):
    """Directly proves app.state.explain_lock (src/api/main.py) serializes
    concurrent /explain requests, independent of whether LIME's actual RNG
    race is triggered on any given run: stubs explain_customer to record how
    many calls are in flight simultaneously, and asserts that number never
    exceeds 1. Fails immediately if the lock is removed."""
    state = {"concurrent": 0, "max_concurrent": 0}
    state_lock = threading.Lock()

    def _slow_explain(_customer, context=None):
        with state_lock:
            state["concurrent"] += 1
            state["max_concurrent"] = max(state["max_concurrent"], state["concurrent"])
        try:
            threading.Event().wait(0.05)
        finally:
            with state_lock:
                state["concurrent"] -= 1
        return {"shap_top_drivers": [], "lime_top_drivers": []}

    monkeypatch.setattr(main, "explain_customer", _slow_explain)

    payload = _customer_payload_stub()
    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(lambda _: client.post("/explain", json=payload), range(6)))

    assert all(r.status_code == 200 for r in responses)
    assert state["max_concurrent"] == 1


def test_explain_value_error_mapped_to_422_with_truncated_detail(client, monkeypatch):
    """Unit-tests main.py's own ValueError -> 422 mapping (Requirement 5)
    directly, since a Pydantic-valid CustomerPayload can no longer trigger
    explain_customer's ValueError paths for most fields now that every
    closed-vocabulary field is Literal-typed (the exception is documented in
    src/api/schemas.py: LIME's label encoders are fit on X_train, a strict
    subset of the Literal vocabularies) -- this exercises that defensive
    layer in isolation rather than leaving it untested."""
    long_message = "unseen category boom " * 50  # > EXPLAIN_ERROR_DETAIL_MAX_LENGTH

    def _raise_value_error(_customer, context=None):
        raise ValueError(long_message)

    monkeypatch.setattr(main, "explain_customer", _raise_value_error)

    response = client.post("/explain", json=_customer_payload_stub())
    assert response.status_code == 422
    assert len(response.json()["detail"]) <= main.EXPLAIN_ERROR_DETAIL_MAX_LENGTH


def test_startup_failure_leaves_health_reachable_and_explain_503(monkeypatch):
    """Uses its own function-scoped monkeypatch and a fresh app (via
    create_app(), never the shared `main.app` singleton the `client`
    fixture already started successfully against) -- a fresh app has its
    own app.state, so this can't corrupt any other test's already-built
    context."""
    def _raise_missing_model(_df):
        raise _missing_model_error()

    monkeypatch.setattr(main, "build_explainer_context", _raise_missing_model)

    with TestClient(main.create_app()) as failed_client:
        health_response = failed_client.get("/health")
        assert health_response.status_code == 200
        body = health_response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False

        explain_response = failed_client.post("/explain", json=_customer_payload_stub())
        assert explain_response.status_code == 503


def test_health_and_explain_error_messages_never_leak_paths(monkeypatch):
    """The startup FileNotFoundError (a realistic, path-bearing message,
    exactly like a genuine missing-metadata-file error) must never reach the
    client -- /health's startup_error and /explain's 503 body are both
    fixed, path-free strings. Compares against the JSON-escaped form of the
    path too, since Windows backslashes are double-escaped in a JSON
    response body and a naive substring check against the raw path would
    silently never match."""
    escaped_path_fragment = json.dumps(str(PROJECT_ROOT))[1:-1]

    def _raise_missing_model(_df):
        raise _missing_model_error()

    monkeypatch.setattr(main, "build_explainer_context", _raise_missing_model)

    with TestClient(main.create_app()) as failed_client:
        health_response = failed_client.get("/health")
        assert str(PROJECT_ROOT) not in health_response.text
        assert escaped_path_fragment not in health_response.text

        explain_response = failed_client.post("/explain", json=_customer_payload_stub())
        assert str(PROJECT_ROOT) not in explain_response.text
        assert escaped_path_fragment not in explain_response.text


# --- /recommend ---------------------------------------------------------------


def test_recommend_returns_expected_shape(client):
    payload = _customer_payload(client, HIGH_RISK_TEST_INDEX, "5178-LMXOP")
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["churn_probability"], float)
    assert isinstance(body["churn_probability_pct"], float)
    assert body["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    assert isinstance(body["actions"], list) and len(body["actions"]) >= 1

    first_action = body["actions"][0]
    assert isinstance(first_action["priority"], int)
    assert isinstance(first_action["action"], str) and first_action["action"]
    assert isinstance(first_action["category"], str) and first_action["category"]
    assert isinstance(first_action["rationale"], str) and first_action["rationale"]
    assert isinstance(first_action["source"], str) and first_action["source"]
    assert first_action["driver_feature"] is None or isinstance(first_action["driver_feature"], str)


def test_recommend_matches_worked_example_critical(client):
    """End-to-end HTTP check on the real 5178-LMXOP customer, against
    whatever model is currently persisted -- same accepted live-model
    dependency as test_explain_matches_verified_high_risk_example, NOT the
    same guarantee as tests/test_action_engine.py's
    test_recommend_actions_matches_worked_example_critical, which asserts
    against a frozen, hardcoded SHAP-driver fixture and survives a retrain.
    This test may need updating (not signal a bug) if a retrain changes
    this customer's driver ranking or risk tier."""
    response = client.post("/recommend", json=CRITICAL_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert body["customerID"] == "5178-LMXOP"
    assert body["risk_tier"] == "Critical"

    actions = body["actions"]
    assert len(actions) == 3
    assert actions[0]["source"] == "tier" and actions[0]["category"] == "escalation"
    assert actions[1]["driver_feature"] == "tenure" and actions[1]["category"] == "onboarding"
    assert actions[1]["expected_churn_reduction_pct"] is None
    assert actions[2]["driver_feature"] == "Contract" and actions[2]["category"] == "contract"
    assert actions[2]["expected_churn_reduction_pct"] == 58.6
    assert not any(a["driver_feature"] == "InternetService" for a in actions)


def test_recommend_matches_worked_example_low(client):
    """End-to-end HTTP check on the real 9763-GRSKD customer, against
    whatever model is currently persisted -- see
    test_recommend_matches_worked_example_critical's docstring for why this
    is a live-model check, not the frozen-fixture guarantee
    tests/test_action_engine.py's own worked-example test has. Note this
    customer is unrelated to LOW_RISK_TEST_INDEX/4376-KFVRS used elsewhere
    in this file; it exists only as this hardcoded raw dict."""
    response = client.post("/recommend", json=LOW_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert body["customerID"] == "9763-GRSKD"
    assert body["risk_tier"] == "Low"

    actions = body["actions"]
    assert len(actions) == 2
    assert actions[0]["source"] == "tier" and actions[0]["category"] == "monitor"
    assert actions[0]["expected_churn_reduction_pct"] is None
    assert actions[1]["driver_feature"] == "Contract" and actions[1]["category"] == "contract"
    assert actions[1]["expected_churn_reduction_pct"] == 14.4


def test_recommend_503_when_calibrated_pipeline_missing(monkeypatch):
    """Independence check (.claude/specs/14-recommend-endpoint.md Req. 2):
    a calibrated-pipeline load failure must 503 only /recommend, leaving
    /health and /explain (which depend only on explainer_context) healthy."""
    def _raise_missing_pipeline(*args, **kwargs):
        raise _missing_model_error()

    monkeypatch.setattr(main.calibration, "load_calibrated_model", _raise_missing_pipeline)

    with TestClient(main.create_app()) as partial_client:
        health_response = partial_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is True

        explain_response = partial_client.post("/explain", json=_customer_payload_stub())
        assert explain_response.status_code == 200

        recommend_response = partial_client.post("/recommend", json=_customer_payload_stub())
        assert recommend_response.status_code == 503
        assert recommend_response.json()["detail"] == main.MODEL_UNAVAILABLE_MESSAGE


def test_recommend_503_when_explainer_context_missing(monkeypatch):
    """Inverse of the above: explainer_context fails to build, calibrated
    pipeline loads fine -- /recommend still 503s (it requires both)."""
    def _raise_missing_model(_df):
        raise _missing_model_error()

    monkeypatch.setattr(main, "build_explainer_context", _raise_missing_model)

    with TestClient(main.create_app()) as partial_client:
        health_response = partial_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is False

        recommend_response = partial_client.post("/recommend", json=_customer_payload_stub())
        assert recommend_response.status_code == 503
        assert recommend_response.json()["detail"] == main.MODEL_UNAVAILABLE_MESSAGE


def test_recommend_503_when_both_artifacts_missing(monkeypatch):
    """Fresh clone / nothing trained yet: both build_explainer_context and
    load_calibrated_model fail -- /health still 200 model_loaded: false,
    both /explain and /recommend 503, matching the spec's stated edge case."""
    def _raise_missing_model(_df):
        raise _missing_model_error()

    def _raise_missing_pipeline(*args, **kwargs):
        raise _missing_model_error()

    monkeypatch.setattr(main, "build_explainer_context", _raise_missing_model)
    monkeypatch.setattr(main.calibration, "load_calibrated_model", _raise_missing_pipeline)

    with TestClient(main.create_app()) as failed_client:
        health_response = failed_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is False

        explain_response = failed_client.post("/explain", json=_customer_payload_stub())
        assert explain_response.status_code == 503

        recommend_response = failed_client.post("/recommend", json=_customer_payload_stub())
        assert recommend_response.status_code == 503


def test_recommend_422_on_unseen_category(client):
    payload = _customer_payload(client, LOW_RISK_TEST_INDEX)
    payload["Contract"] = "Lifetime"
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    assert "Lifetime" not in response.text


def test_recommend_and_explain_share_lock_under_concurrency(client, monkeypatch):
    """Directly proves /explain and /recommend serialize on the SAME lock
    (.claude/specs/14-recommend-endpoint.md Requirement 5) -- a per-route
    lock would allow same-route concurrency of 1 but cross-route
    concurrency of 2; interleaving both routes across 6 threads and still
    observing max_concurrent == 1 rules that out.

    max_concurrent == 1 alone can't distinguish "the lock serialized these"
    from "the 6 requests just never happened to overlap" -- the wall-clock
    assertion below is the positive control: each stub holds its slot for
    HOLD_SECONDS, so 6 fully-serialized calls take at least
    5 * HOLD_SECONDS (allowing one to run without waiting), while 6 calls
    with real concurrency would finish far faster.
    """
    HOLD_SECONDS = 0.05
    state = {"concurrent": 0, "max_concurrent": 0, "calls": 0}
    state_lock = threading.Lock()

    def _bump_and_hold():
        with state_lock:
            state["concurrent"] += 1
            state["calls"] += 1
            state["max_concurrent"] = max(state["max_concurrent"], state["concurrent"])
        try:
            threading.Event().wait(HOLD_SECONDS)
        finally:
            with state_lock:
                state["concurrent"] -= 1

    def _slow_explain(_customer, context=None):
        _bump_and_hold()
        return {"shap_top_drivers": [], "lime_top_drivers": []}

    def _slow_recommend(_customer, pipeline=None, explainer_context=None, top_n=3):
        _bump_and_hold()
        return {"churn_probability": 0.5, "churn_probability_pct": 50.0, "risk_tier": "Medium", "actions": []}

    monkeypatch.setattr(main, "explain_customer", _slow_explain)
    monkeypatch.setattr(main, "recommend_actions_for_customer", _slow_recommend)

    payload = _customer_payload_stub()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(client.post, "/explain" if i % 2 == 0 else "/recommend", json=payload)
            for i in range(6)
        ]
        responses = [f.result() for f in futures]
    elapsed = time.monotonic() - start

    assert all(r.status_code == 200 for r in responses)
    assert state["calls"] == 6
    assert state["max_concurrent"] == 1
    assert elapsed >= 5 * HOLD_SECONDS
