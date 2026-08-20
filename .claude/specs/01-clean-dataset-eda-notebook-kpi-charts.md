# Spec + Plan: Clean Dataset Load + EDA Notebook + KPI Charts (Phase 1)

> Location note: this file lives at `.claude/specs/` per CLAUDE.md §4's repo
> tree, which marks `.claude` as "claude.md and also Specs folder" — not the
> generic top-level `specs/` directory. Spec and plan are combined in one file
> since the feature was requested at a single file path.

---

## PART 1 — SPEC

### Feature

Formalize Phase 1 of RetainIQ — the data-cleaning module (`src/data/`), the EDA
notebook (`notebooks/01_eda.ipynb`), the KPI chart generator, and the pytest
guardrails — by fixing the real defects research turned up (a `.gitignore` bug
that has silently kept all Phase 1 source code out of git, and a missing
runtime dependency), tightening test coverage to the CLAUDE.md §9 bar, and
bringing the phase tracker in line with reality.

### Problem / motivation

Phase 1's logic is already written and its output already validates correctly
against every signal CLAUDE.md §6 requires (verified below), but three concrete
problems were found on close inspection and must be fixed before Phase 1 can
honestly be called "done":

1. **`.gitignore` is silently excluding real source code, not just data
   artifacts.** `.gitignore` currently has unanchored `data/` and `models/`
   entries. Git matches an unanchored pattern against a directory of that name
   at *any* depth, so it also matches `src/data/` and `src/models/` — not just
   the top-level `data/` and `models/` that CLAUDE.md §7 says to ignore.
   Verified directly:
   ```
   $ git check-ignore -v src/data/config.py src/data/load_data.py src/data/eda.py src/data/__init__.py src/models/__init__.py
   .gitignore:1:data/	src/data/config.py
   .gitignore:1:data/	src/data/load_data.py
   .gitignore:1:data/	src/data/eda.py
   .gitignore:1:data/	src/data/__init__.py
   .gitignore:2:models/	src/models/__init__.py
   ```
   `git ls-files` confirms only the Phase-0 scaffold (`src/__init__.py`,
   `src/api/__init__.py`, `src/explain/__init__.py`, `src/features/__init__.py`,
   `src/recommend/__init__.py`, plus root config files) has ever been committed.
   None of Phase 1's actual code — `load_data.py`, `eda.py`, `config.py` — is in
   git history. Every future phase that adds files under `src/data/` or
   `src/models/` would silently fail to commit the same way.

2. **`requirements.txt` is missing dependencies the code actually imports.**
   `src/data/eda.py` does `import matplotlib.pyplot as plt` directly, and
   `notebooks/01_eda.ipynb` does `from IPython.display import Image, display`
   and requires a Jupyter/IPython kernel to execute at all. Checked against the
   installed environment (`pip freeze`): `matplotlib`, `jupyter`, `ipykernel`,
   and `nbformat` are present locally, but none of them appear in
   `requirements.txt` — only `seaborn` is listed, and while seaborn pulls in
   matplotlib as a transitive dependency, `pip install -r requirements.txt` on
   a clean machine would still have no Jupyter kernel to run the notebook with.
   This breaks the Setup instructions in `README.md`
   (`pip install -r requirements.txt` then "run the notebook") for anyone
   without the current machine's preexisting global packages.

3. **The phase tracker and README status are stale.** CLAUDE.md §14 still shows
   Phase 1 as `☐`, and `README.md`'s Status section still says "Phase 0
   (scaffold) complete," even though Phase 1's logic, notebook, and tests exist
   and pass. Anyone reading either file — including a recruiter — gets an
   inaccurate picture of project state.

**What's already correct and does not need touching:** `clean_data()`'s logic
(blank `TotalCharges` → 0, `Churn` → binary, `SeniorCitizen` → Yes/No,
`customerID` dropped); all 6 existing tests in `tests/test_data.py`; every
`eda.py` chart function; the notebook's analysis and its match against the
CLAUDE.md §6 signal table (contract 42.71%/11.27%/2.83%, electronic-check
45.29%, fiber-optic 41.89%, no-tech-support 41.64%, tenure 18.0 vs 37.6 months
churned/retained — all reproduced and confirmed by direct execution). One
borderline pattern was investigated and found *not* to be a bug:
`plot_tenure_distribution()` in `eda.py` builds a seaborn legend from
`hue=TARGET_COLUMN` (an int column, palette `{0: blue, 1: red}`) and then calls
`ax.legend(title="Churn", labels=["Yes", "No"])` to override the default "0"/"1"
tick text. This looks at first glance like it could mislabel the swatches, but
empirically re-creating the exact call (`ax.legend()` after a stacked
`histplot`) shows matplotlib rebuilds handle order from the stacked-patch
z-order on the second call, and the override lands correctly: "Yes" → red
(1), "No" → blue (0). It is fragile — it depends on matplotlib's legend
handle-reordering behavior on a stacked histogram rather than an explicit
`hue_order` — but it is not presently wrong, so it is listed as an optional
hardening item below, not a required fix.

### Goals / non-goals

**Goals**
- Fix `.gitignore` so `data/` and `models/` are anchored to the project root
  (`/data/`, `/models/`) and no longer shadow `src/data/`/`src/models/`.
- Commit Phase 1's real source (`src/data/*.py`, `notebooks/01_eda.ipynb`,
  `tests/test_data.py`, `reports/figures/*.png`) to git.
- Add `matplotlib`, `jupyter`, and `ipykernel` to `requirements.txt` so
  `pip install -r requirements.txt` alone is sufficient to run
  `python -m src.data.eda` and execute the notebook end-to-end.
- Add the schema/dtype test CLAUDE.md §9 calls for and that today isn't fully
  covered: cleaned data's column set and target dtype.
- Update CLAUDE.md §14 (Phase 1 row) and README's Status section to match
  reality.

**Non-goals**
- No changes to `src/features/`, `src/models/`, `src/explain/`,
  `src/recommend/`, the API, or the dashboard — later phases.
- No new charts or EDA questions — existing analysis already covers every
  CLAUDE.md §6 signal.
- No change to `clean_data()`'s behavior.
- No rewrite of `plot_tenure_distribution()`'s legend handling — flagged as
  optional hardening (see Functional Requirement 8), not required, since it is
  currently correct.
- Not touching the top-level `CLAUDE.md` file's deleted status in git (working
  tree shows `D CLAUDE.md` versus HEAD) — that predates this feature and its
  disposition (intentional move to `.claude/CLAUDE.md`, or accidental) is the
  user's call, not something to silently resolve here.

### User stories

- As the **engineer (Priyabrata)**, I want Phase 1's actual code committed to
  git so Phase 2 builds on a real, recoverable history instead of files that
  only exist in the working tree.
- As the **engineer**, running `pip install -r requirements.txt` on a fresh
  clone, I want that single command to be enough to run the EDA notebook and
  regenerate the KPI charts, without needing to guess which packages a global
  environment happened to already have installed.
- As a **recruiter/reviewer** skimming the repo, I want CLAUDE.md and README to
  say Phase 1 is done, backed by passing tests, so I can trust the phase
  tracker without re-deriving status myself.
- As the **engineer**, I want a schema-shape test on the cleaned data so a
  future Kaggle re-download or upstream column change fails loudly in CI
  instead of silently breaking Phase 2's feature pipeline.

### Functional requirements

1. `.gitignore`'s `data/` and `models/` entries MUST become `/data/` and
   `/models/`. Verify with
   `git check-ignore -v src/data/config.py src/data/load_data.py src/data/eda.py src/data/__init__.py src/models/__init__.py`
   returning no output (no longer ignored), while
   `git check-ignore -v data/raw/telco.csv` still reports it ignored.
2. After the `.gitignore` fix, `src/data/__init__.py`, `src/data/config.py`,
   `src/data/load_data.py`, `src/data/eda.py`, `notebooks/01_eda.ipynb`,
   `tests/test_data.py`, and `reports/figures/*.png` MUST be stageable and
   committed. `src/data/__pycache__/` MUST remain ignored (already correctly
   matched by the separate `__pycache__/` line).
3. `requirements.txt` MUST add `matplotlib`, `jupyter`, and `ipykernel` (kept
   unpinned, consistent with every other entry in the file). No other line in
   `requirements.txt` changes.
4. A new test in `tests/test_data.py` MUST assert
   `set(clean_df.columns) == EXPECTED_COLUMNS`, where `EXPECTED_COLUMNS` is the
   verified 20-column set (`gender`, `SeniorCitizen`, `Partner`, `Dependents`,
   `tenure`, `PhoneService`, `MultipleLines`, `InternetService`,
   `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
   `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`,
   `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`) — i.e. the 21 raw
   columns minus `customerID`.
5. A new test MUST assert `clean_df[TARGET_COLUMN].dtype == np.int64` (verified
   actual dtype via direct execution — `.astype(int)` on this platform produces
   `int64`, not a generic Python `int` object column).
6. CLAUDE.md §14's Phase 1 row MUST change from `☐` to a completed marker
   consistent with the table's existing style.
7. README's Status section MUST state Phase 1 (data + EDA) is complete,
   preserving the existing pointer to CLAUDE.md §14.
8. *(Optional hardening, not required for done)* `plot_tenure_distribution()`
   MAY be changed to pass explicit `hue_order=[0, 1]` to `histplot` and build
   legend labels from a `{0: "No", 1: "Yes"}` mapping instead of relying on
   matplotlib's post-hoc handle reordering, so the correct legend mapping is
   guaranteed by code rather than incidental draw order. Do this only if time
   allows; it is not part of Definition of Done.
9. None of the above may change `clean_data()`'s behavior or any existing
   chart's output — all 6 current tests plus the 2 new tests must pass
   unmodified.

### Data & model impact

None to the model or preprocessing pipeline. This feature only fixes git
tracking, a missing dependency declaration, documentation, and adds two
guardrail tests. `clean_data()`'s output (shape, dtypes, values) is unchanged.

### ML guardrails (mandatory check)

N/A — no feature engineering, splitting, resampling, or model training happens
in this feature. Confirmed no regression: the two new tests only assert column
membership and dtype on the already-cleaned frame; they touch no leakage-prone
path. Existing guardrails already respected by the code being formalized:
`Churn` is mapped to binary and is not fed back in as a feature; no
probability or post-outcome column exists in the cleaned frame;
`random_state=42` isn't yet applicable (no splitting occurs until Phase 2).

### API / UI surface

None — no FastAPI endpoint or Streamlit view touched.

### Edge cases & failure states

- **`.gitignore` anchoring accidentally un-ignores something else named
  `data` or `models` elsewhere in the tree**: guarded by Functional
  Requirement 1's explicit before/after `git check-ignore -v` checks on both
  `src/data/...` (should flip to tracked) and `data/raw/telco.csv` (should stay
  ignored). A repo-wide `find . -type d -iname data -o -type d -iname models`
  should be run once during implementation to confirm there is no third
  directory of either name that the anchoring change would affect.
- **Adding `jupyter`/`ipykernel` to `requirements.txt` pulls in a heavy
  dependency tree**: acceptable — the notebook cannot run without a kernel
  regardless, and CLAUDE.md's own repo structure documents `notebooks/` as a
  first-class directory, so the dependency is legitimate, not incidental.
- **Committing `reports/figures/*.png` now, then regenerating them later with
  visual diffs**: acceptable at Phase 1 size (10 small PNGs); revisit only if
  it becomes a noisy diff pattern in a later phase.
- **New schema test is intentionally brittle** if a future Kaggle re-download
  adds, renames, or removes a column — that is the desired behavior: fail
  loudly rather than let a silent shape change flow into Phase 2.

### Security notes

None — no new untrusted input, secret, or externally-reachable dependency.
`jupyter`/`ipykernel`/`matplotlib` are widely-used, already-present-locally
packages being formally declared, not newly introduced attack surface. Verify
during implementation that `.env` remains ignored after the `.gitignore` edit
(`git check-ignore -v .env`) since it is a separate, unrelated line but worth
a one-line confirmation given the file is being touched.

### Success criteria

- `git check-ignore -v src/data/config.py src/data/load_data.py src/data/eda.py src/models/__init__.py`
  returns no output.
- `git log --oneline -- src/data/load_data.py` shows a commit.
- `pip install -r requirements.txt` on a clean environment is sufficient to run
  `python -m src.data.eda` and execute `notebooks/01_eda.ipynb` top to bottom.
- `pytest -q` passes, 8/8 (6 existing + 2 new).
- CLAUDE.md §14 shows Phase 1 complete; README Status section reflects it.
- `quality-reviewer` and `security-reviewer` report no unresolved findings on
  the diff.

### Out of scope

- Phase 2 (feature pipeline, model training, MLflow) — not started here.
- Any change to the cleaning/EDA logic itself.
- The optional legend-hardening in Functional Requirement 8.
- Resolving the pre-existing `D CLAUDE.md` (top-level, vs. `.claude/CLAUDE.md`)
  git status — out of scope for this feature; flag to the user separately.

---

## PART 2 — PLAN

### Approach

Fix the `.gitignore` root cause first, verify it with `git check-ignore`, then
commit the now-visible Phase 1 source as one commit. Follow with a second
commit for the dependency, test, and documentation additions — kept separate
because the `.gitignore`/source commit is the urgent, revertible-on-its-own fix
(unblocks Phase 2 from silently losing work), while the docs/test commit is
lower-stakes polish that shouldn't gate the first.

**Alternative rejected:** `git add -f` to force-add the currently-ignored files
without editing `.gitignore`. Rejected because it treats the symptom, not the
cause — every future file added under `src/data/` or `src/models/` would need
the same manual `-f`, and it would eventually be forgotten, silently repeating
today's problem.

### Task breakdown

- [ ] **1. Fix `.gitignore` anchoring** — edit `.gitignore`: `data/` → `/data/`,
      `models/` → `/models/`. No other line changes.
- [ ] **2. Verify the fix** — run
      `git check-ignore -v src/data/config.py src/data/load_data.py src/data/eda.py src/data/__init__.py src/models/__init__.py`
      (expect nothing) and `git check-ignore -v data/raw/telco.csv .env`
      (expect both still ignored).
- [ ] **3. Commit Phase 1 source** — review `git status` output first (confirm
      no `data/raw/*` or `__pycache__/` slipped in), then
      `git add src/data/ notebooks/01_eda.ipynb tests/test_data.py reports/figures/`
      and commit as
      `phase 1: data loading, cleaning, EDA notebook, KPI charts`.
- [ ] **4. Add missing runtime deps** — in `requirements.txt`, add `matplotlib`,
      `jupyter`, `ipykernel` (unpinned, matching existing style), placed near
      `seaborn`/`plotly` in the viz section.
- [ ] **5. Add the schema test** — in `tests/test_data.py`, add
      `test_expected_columns_present(clean_df)` asserting the exact 20-column
      set from spec Functional Requirement 4, and
      `test_churn_column_is_int64_dtype(clean_df)` asserting
      `clean_df[TARGET_COLUMN].dtype == np.int64`.
- [ ] **6. Update CLAUDE.md §14 phase tracker** — mark the Phase 1 row done.
- [ ] **7. Update README Status section** — replace "Phase 0 (scaffold)
      complete" with a line noting Phase 1 (data + EDA) is also complete,
      keeping the "See CLAUDE.md §14" pointer.
- [ ] **8. Commit deps/test/docs additions** —
      `git add requirements.txt tests/test_data.py CLAUDE.md README.md` (only
      `.claude/CLAUDE.md` if that is confirmed to be the live copy — do not
      touch the top-level `CLAUDE.md` deletion without the user's explicit
      confirmation per spec's Out of scope), commit as
      `phase 1: pin notebook/EDA deps, add schema test, update status`.

### Tests to write (hand to test-writer)

- `test_expected_columns_present` — cleaned frame's column set matches the
  documented 20-column schema exactly (catches added/renamed/dropped columns).
- `test_churn_column_is_int64_dtype` — `Churn` column dtype is `int64` after
  cleaning (catches a future pandas/mapping change silently producing floats or
  object dtype).

Both belong in `tests/test_data.py`, reusing the existing module-scoped
`clean_df` fixture.

### Quality gates

1. **test-runner** — run `pytest -q`, confirm 8/8 pass, fix any straightforward
   regression.
2. **quality-reviewer** — review the `.gitignore` diff, new test assertions,
   `requirements.txt` addition, and CLAUDE.md/README doc edits.
3. **security-reviewer** — confirm no secret/credential file becomes
   accidentally trackable as a side effect of the anchoring change; spot-check
   `git check-ignore -v .env` still reports ignored.
4. Commit only after both reviews are clean.

### Risks / rollback

- **Risk:** anchoring `data/` → `/data/` unignores something unintended
  elsewhere. **Mitigation:** Task 2's explicit `git check-ignore -v` checks
  before any `git add`, plus the repo-wide `find` sanity check from the spec's
  edge cases.
- **Risk:** adding `jupyter`/`ipykernel` to `requirements.txt` significantly
  slows a fresh `pip install`. **Mitigation:** acceptable trade-off — the
  notebook is unusable without them; no action needed.
- **Rollback:** each commit is small and separable. If Task 5's schema test or
  Task 4's dependency addition needs rework, `git revert` the second commit
  only — the first commit (source + `.gitignore` fix) stays intact and Phase 2
  can still build on it.

### Definition of done

- All 8 tasks checked off.
- `pytest -q` green (8/8).
- `git log --oneline -- src/data/load_data.py` shows Phase 1 source in history.
- `git check-ignore -v` confirms the anchoring fix without over- or
  under-ignoring.
- `requirements.txt` includes `matplotlib`, `jupyter`, `ipykernel`.
- CLAUDE.md §14 and README Status both reflect Phase 1 complete.
- `quality-reviewer` and `security-reviewer` report no unresolved findings.
- All Success Criteria in Part 1 are met.
