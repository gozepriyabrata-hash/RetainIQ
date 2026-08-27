"""RetainIQ API: POST /explain (CLAUDE.md Section 10) + GET /health.

Thin FastAPI layer only -- every request is validated by Pydantic
(src.api.schemas) and dispatched straight to
src.explain.local_explainer.explain_customer(); no explanation, SHAP, LIME,
scoring, or risk-tier logic is reimplemented here (CLAUDE.md Section 4).

The explainer context (a SHAP TreeExplainer + LimeTabularExplainer built
against the persisted production model) is built once at startup via
`lifespan`, not per request (CLAUDE.md Section 10), and stashed on
app.state. A missing/stale model at startup (models/ is git-ignored per
CLAUDE.md Section 7, so a fresh clone has none until `python -m
src.models.train` + `python -m src.models.calibration` are run) does not
crash the process: /health reports model_loaded: false and /explain
returns 503 until a real model is trained.

See .claude/specs/12-explain-endpoint.md for the full spec.
"""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import CustomerPayload, ExplainResponse, HealthResponse
from src.data.load_data import load_clean_data
from src.explain.local_explainer import build_explainer_context, explain_customer

logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE_MESSAGE = (
    "Model is not available. Run `python -m src.models.train` and "
    "`python -m src.models.calibration` first."
)
# explain_customer's ValueError already re-quotes the offending request value
# (e.g. an unseen category) into its message; truncated defensively before it
# reaches the response body, on top of Literal-typed fields already rejecting
# most such values before explain_customer ever runs.
EXPLAIN_ERROR_DETAIL_MAX_LENGTH = 500


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the explainer context once; never let a missing/broken model crash startup.

    Catches broadly (not just the exception types build_explainer_context's
    own docstring names) because a corrupted or version-mismatched pickle can
    surface as pickle.UnpicklingError, EOFError, AttributeError, or an
    xgboost-internal error depending on exactly how it's broken -- this is a
    startup-degradation boundary, not ordinary control flow, and the whole
    point is that no failure here should crash the process (mirrors
    app/dashboard.py's broad-but-intentional system-boundary handling).
    """
    try:
        app.state.explainer_context = build_explainer_context(load_clean_data())
        app.state.startup_error = None
    except Exception:
        logger.exception("Failed to build explainer context at startup.")
        app.state.explainer_context = None
        app.state.startup_error = "Model artifacts unavailable at startup."
    yield


def create_app() -> FastAPI:
    """Build a fresh FastAPI app instance.

    A factory (rather than a single module-level app touched directly by
    every route) so a caller -- in particular a test exercising the
    startup-failure path -- can construct an independent app/app.state
    without mutating the shared `app` singleton every other test relies on.
    """
    app = FastAPI(title="RetainIQ API", lifespan=lifespan)
    # Set before lifespan ever runs (not just inside it) so /health is a
    # genuine "200 always" liveness check even if hit before startup
    # completes -- Starlette's State raises AttributeError on an unset key,
    # which would otherwise turn an early health check into a bare 500.
    app.state.explainer_context = None
    app.state.startup_error = None
    # LimeTabularExplainer mutates its own numpy RandomState on every
    # explain_instance() call; FastAPI dispatches sync `def` routes to a
    # threadpool, so concurrent /explain requests would otherwise race on
    # that shared, non-thread-safe state. One lock serializes the (already
    # fast, ~0.25s) explain_customer call rather than building a
    # LimeTabularExplainer per request, which would defeat the whole
    # load-once-at-startup design.
    app.state.explain_lock = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Strip the attacker-controlled `input` field from Pydantic's default
        422 body (FastAPI's default handler echoes the full rejected value
        back unbounded, e.g. a multi-megabyte string sent for a Literal-typed
        field) -- loc/msg/type are kept, so which field failed and why is
        still reported.
        """
        errors = exc.errors()
        for error in errors:
            error.pop("input", None)
            error.pop("url", None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(errors)},
        )

    # No blocking work -- an `async def` route runs directly on the event
    # loop instead of Starlette's shared sync-route threadpool, so /health
    # stays responsive even if every threadpool worker is queued behind
    # app.state.explain_lock during a burst of /explain calls.
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Liveness/readiness check -- always 200, reports whether a model is loaded."""
        return HealthResponse(
            status="ok",
            model_loaded=request.app.state.explainer_context is not None,
            startup_error=request.app.state.startup_error,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(payload: CustomerPayload, request: Request) -> ExplainResponse:
        """Customer -> SHAP + LIME top-3 churn drivers (CLAUDE.md Section 10)."""
        context = request.app.state.explainer_context
        if context is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, MODEL_UNAVAILABLE_MESSAGE)

        try:
            with request.app.state.explain_lock:
                result = explain_customer(payload.to_customer_dict(), context=context)
        except ValueError as exc:
            detail = str(exc)[:EXPLAIN_ERROR_DETAIL_MAX_LENGTH]
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail) from exc
        except Exception:
            # Not expected to be reachable (Literal-typed fields already
            # reject most bad input, and explain_customer's own failure
            # modes are ValueError) -- logged here so an unanticipated
            # failure is still visible server-side; FastAPI's default
            # handler already returns a safe, generic 500 with no traceback.
            logger.exception("Unexpected error while explaining a customer.")
            raise

        return ExplainResponse(**result)

    return app


app = create_app()
