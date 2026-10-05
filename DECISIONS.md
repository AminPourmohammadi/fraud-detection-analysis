
## 2026-09-30: notebook 01 opening (Amin)

- Open with an explanation of why this dataset, including what it cannot do.
- Remove the "what makes this more than a tutorial" section.
- Remove the "how to read it" section.

## 2026-09-30: notebook 01, section 5.3 (Amin)

- Show the code that builds the feature list, not only the text saying which
  columns are kept and dropped. Built it as "all columns minus the dropped
  ones" with each reason written next to the code, and added a check that it
  matches the list the API imports.

## 2026-10-01: why a time split, not shuffled or stratified (Amin's question)

Amin asked whether the different fraud rates in the train and test periods
are a concern, whether we can do anything, and why we did not shuffle or
stratify. Checked on the data; answers added to notebook 01 (sections 6.1,
6.2, 12.1, 12.6, 12.7).

- Rate gap: 59% of it is because training holds both nights (0.42% fraud)
  and the later parts are daytime only (training daytime 0.15%, validation
  0.095%, test 0.127%). The rest is unexplained: a real change or day-to-day
  variation; two days cannot tell.
- The model's average prediction matched the actual rate on validation
  (0.105% vs 0.095%) and on test (0.128% vs 0.127%).
- Shuffled and stratified splits: PR-AUC 0.84 and 0.83 against 0.80 for the
  time split, ROC-AUC 0.98 for all three. The gap is within seed noise.
  Shuffled splits leave near-copies of test frauds in training (7.7% against
  0%).
- Decision 4 (time split) stays, now with evidence. Status: to review.
- Also corrected a sentence in section 3.4 that overstated the night-time
  fraud rate ("1% to 2%"; only some single hours reach that).

## 2026-10-02: precision in the top 100 (Amin)

Amin edited sections 6 and 6.1 himself, then asked to change the notebook
from section 7 onward and add a top-100 precision metric.

- Added Precision@100 (share of the 100 highest-scored payments that are
  fraud). Explained in section 7, with a function and a printout of the best
  possible score (0.59 on validation, 0.76 on test, because there are only 59
  and 76 frauds).
- Added as a column in the validation comparison (section 10), the test
  ranking (12.1) and the split comparison (12.6), with a reading sentence
  each, and one line in the recap.
- Result: plain XGBoost holds 47 of 59 validation frauds and 60 of 76 test
  frauds in its top 100. Gaps between models are one to three frauds, so the
  metric does not rank them; PR-AUC does a bit better.
- Fixed the reference to "section 6.2" in 12.6, which Amin removed; it now says
  "section 6".
- Not changed: his edits to sections 6 and 6.1, the README, the saved model
  and API files.

## 2026-10-02: does tuning help? (Amin)

Amin asked whether the XGBoost model should be tuned before the rest of the
analysis, and said to go ahead with a light check on his saved notebook.

- Added section 9.4. Tuned inside the training part with three time-ordered
  folds (train on hours 0 to 14, 20, 26; score the next six hours), not on
  validation, which has 59 frauds and is needed for the decision rule.
- Twelve settings (depth 3/4/6, rate 0.05/0.1, 200/400 trees), scored by
  PR-AUC. Rule written before looking: switch only if the best beats the
  defaults by more than the spread between folds.
- Result: all 12 between 0.783 and 0.811, fold spread about 0.08. Best
  (depth 3, rate 0.05, 400 trees) beats the defaults by 0.009. Defaults kept.
  An honest dead end, noted in the notebook and the recap.
- `make_xgb` now takes depth, rate and trees (defaults unchanged), so the
  tuning loop uses the same model code. Nothing downstream changed.
- Kept your edits to sections 6, 7, 10, 10.1 and 11.

## 2026-10-03: show how the helper functions are built (Amin)

Amin said he did not know how `build_reference` was built, asked for the
skill to require showing every function's construction, and then asked to
update the notebook too.

- Notebook 01 now shows the source of every helper it imports from `app/`,
  with a short explanation before and after, and a small call on real or
  example values: `add_hour_of_day` (5.1), `expected_loss` and
  `flag_for_review` (end of 11.1), and `make_edges`, `share_per_bin` and
  `build_reference` (new 14.1, with a worked example on `Amount`).
- The code stays in `app/features.py` and `app/monitoring.py`, because the
  API and the drift script import the same functions. The notebook prints
  the source with `inspect.getsource`, so there is one copy.
- Imports in the first code cell now include `inspect`, `expected_loss`,
  `make_edges` and `share_per_bin`. No results changed.
- Not done: notebook 02 also uses helpers from `app/monitoring.py` (`psi`,
  `drift_report`); offered, not asked for.

## 2026-10-05: new project without the API (Amin)

Amin read both notebooks but had not used FastAPI before, and the API code
could look like it was written by someone else. He chose to leave it out for
now: put the notebooks in a new folder with their own README, and keep the
API project as it is.

- New folder: `fraud-detection-analysis`. It holds notebooks 01 and 02,
  `src/features.py` and `src/monitoring.py` (the same helpers as `app/`,
  renamed), the model and model card, the charts, a new README, `LICENSE`,
  `requirements.txt` and `.gitignore`. Both notebooks were re-run there; all
  results are unchanged (test PR-AUC 0.7968, costs €2,807 / €4,006 /
  €3,973 / €8,395).
- Left in `fraud-detection-api`, untouched: `app/main.py`, the tests, the
  Dockerfile, the CI workflow, `scripts/check_drift.py`, the example
  requests and the old README.
- Notebook changes: imports now come from `src/`; text that mentioned the
  API or the command-line script was reworded; the "example requests" cells
  at the end of notebook 01 were removed; notebook 02's section 8 now only
  shows the recommended report.
- This file's earlier entries came from the API project and mention it.
  They are kept as the history of the choices.
- Amin may come back to the API later, once he has learned FastAPI.
