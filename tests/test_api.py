import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from httpx import Response

import src.api.main as main
from src.api.schemas import CUSTOMER_ID_MAX_LENGTH, MAX_MONTHLY_CHARGES, MAX_TENURE_MONTHS
from src.data.config import PROJECT_ROOT
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
