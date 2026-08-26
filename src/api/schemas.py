"""Pydantic request/response models for the POST /explain contract (CLAUDE.md
Section 10). CustomerPayload covers every raw Telco column CLAUDE.md Section 6
documents except Churn; ShapDriver/LimeDriver/ExplainResponse mirror
src.explain.local_explainer.explain_customer's existing dict output verbatim
-- no reshaping. See .claude/specs/12-explain-endpoint.md for the full spec.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed-vocabulary columns are typed as Literal (not bare str) so an unseen
# category is rejected at the validation boundary -- FastAPI's own 422, not a
# later library-specific error surfaced through explain_customer -- and so
# /docs documents the real allowed values instead of "any string." Verified
# directly against load_clean_data()'s actual unique values per column (the
# full ~7,043-row dataset). explain_customer's own ValueError path stays in
# place behind this (src/api/main.py) rather than becoming dead code: LIME's
# label encoders are fit on X_train only (the 80% split, not the full
# dataset), so a category present in the full data but absent from that
# split would still pass this Literal check and only fail one layer deeper.
_YES_NO = Literal["Yes", "No"]
_YES_NO_NO_PHONE = Literal["Yes", "No", "No phone service"]
_YES_NO_NO_INTERNET = Literal["Yes", "No", "No internet service"]

CUSTOMER_ID_MAX_LENGTH = 64
# Sanity bounds, not business rules -- generous enough to admit any real
# customer while capping the unbounded-string/number DoS surface a bare
# numeric lower-bound-only field would otherwise leave open.
MAX_TENURE_MONTHS = 1200  # 100 years
MAX_MONTHLY_CHARGES = 100_000.0
MAX_TOTAL_CHARGES = 10_000_000.0


class CustomerPayload(BaseModel):
    """Raw customer attributes, the same shape explain_customer() expects.

    SeniorCitizen accepts both the raw CSV's native 0/1 int encoding and the
    "Yes"/"No" string form clean_data() normalizes it to: _clean_common_fields
    (src/data/load_data.py) maps 0/1 -> Yes/No only when the incoming dtype
    isn't already object, so both forms already flow correctly into
    explain_customer -- this schema accepts both rather than picking one and
    rejecting a caller sending the dataset's own native encoding.
    """

    # extra="forbid" is a leakage guard, not just strictness: it rejects a
    # raw Telco row carrying Churn (or any other stray field) rather than
    # silently dropping it -- keep this "forbid", never relax to "ignore".
    model_config = ConfigDict(extra="forbid")

    customerID: str | None = Field(default=None, max_length=CUSTOMER_ID_MAX_LENGTH)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1, "Yes", "No"]
    Partner: _YES_NO
    Dependents: _YES_NO
    tenure: int = Field(ge=0, le=MAX_TENURE_MONTHS)
    PhoneService: _YES_NO
    MultipleLines: _YES_NO_NO_PHONE
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: _YES_NO_NO_INTERNET
    OnlineBackup: _YES_NO_NO_INTERNET
    DeviceProtection: _YES_NO_NO_INTERNET
    TechSupport: _YES_NO_NO_INTERNET
    StreamingTV: _YES_NO_NO_INTERNET
    StreamingMovies: _YES_NO_NO_INTERNET
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: _YES_NO
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0, le=MAX_MONTHLY_CHARGES, allow_inf_nan=False)
    # None means "not yet billed" (CLAUDE.md Sec 6's 11 blank-TotalCharges
    # new customers) -- to_customer_dict() normalizes this to 0.0, matching
    # clean_data()'s own imputation, rather than passing None onward.
    TotalCharges: float | None = Field(default=None, ge=0, le=MAX_TOTAL_CHARGES, allow_inf_nan=False)

    def to_customer_dict(self) -> dict[str, object]:
        """Flat dict matching explain_customer's expected shape.

        customerID is omitted entirely when absent (explain_customer treats
        a present customerID as pass-through, not a required feature column).
        """
        data = self.model_dump(exclude_none=True)
        data["TotalCharges"] = self.TotalCharges if self.TotalCharges is not None else 0.0
        return data


class ShapDriver(BaseModel):
    """One entry of explain_customer()'s shap_top_drivers list."""

    feature: str
    customer_value: str | int | float
    shap_value: float
    direction: str
    reason: str


class LimeDriver(BaseModel):
    """One entry of explain_customer()'s lime_top_drivers list."""

    feature: str
    lime_weight: float
    direction: str
    reason: str


class ExplainResponse(BaseModel):
    """POST /explain's response body -- field-for-field identical to
    explain_customer()'s dict output."""

    customerID: str | None = None
    shap_top_drivers: list[ShapDriver]
    lime_top_drivers: list[LimeDriver]


class HealthResponse(BaseModel):
    """GET /health's response body."""

    status: str
    model_loaded: bool
    startup_error: str | None = None
