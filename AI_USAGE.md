# AI assistance log

## 2026-11-16 — Codex

Tools: Codex assistant, local shell/Python, web lookup.
Prompt summary: implement all coursework Python modules, tests, experiment scripts,
analysis utilities and ordered run instructions using the specification reproduced
in chat. Respect informant Algorithm 39, training-only fitness, reproducible seeds
and no AI-run/interpreted coursework experiments or AI-written report. The full
original prompt is retained in the chat; this entry is a summary.

Changes: activation registry/stable logistic, configurable ANN and legacy API
wrapper, portable train-only data scaler, binary metrics, training fitness,
informant PSO, boundary and encoded-activation extensions, seeded experiment
runner, training-only selection, metadata/artifacts, student-run analysis,
correctness tests and README. Exact A–I architectures remain a student input.

Verification: correctness tests including sphere optimisation, existing exercise
scripts, and synthetic integration checks. No Banknote training or coursework
result interpretation was performed. Initial test invocation from the parent
folder failed imports; documented commands run from inside the project folder.

Reference verified: Luke, Essentials of Metaheuristics, Algorithm 39, second
edition: https://people.cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf .
The official coursework PDF and Week 7 slides were unavailable.

Students must review/understand code, enter actual A–I settings, run experiments,
interpret results, research comparisons, write the report and sign contributions.
Add your actual changes and future prompts below; AI cannot attest to student work.

Environment follow-up: installed missing matplotlib and its dependencies from
requirements.txt in the existing venv. Synthetic four-row runner checks passed
for both fixed and particle-encoded activations. No Banknote training was used.


## Disk-full recovery assistance
Prompt: student reported Errno 28 while saving exp1 seed artifacts.
Changes: added explicit --resume with metadata/config/seed validation and
completed-artifact integrity checks; atomic NPZ/CSV/metadata replacement; updated
run instructions. Existing experiment files were preserved. Verification uses
mocked runs and temporary synthetic files only; no coursework experiments run
or interpreted. Disk availability and results directory size were inspected.

## 2026-11-24 — implementation audit fixes
Prompt summary: student requested implementation cleanup and guidance on where
visualisations could explain Tasks 1–5. No report text or experimental
interpretation requested/generated as part of the fixes.
Changes: metadata-backed complete-suite validation shared by selection and
analysis; explicit opt-in for paired t-tests and corrected statistical guidance;
removed outdated mandatory A–I/PDF-unavailable claims from README and SPEC;
added focused boundary, inertia, artifact reconstruction and validation tests.
Saved the current installed environment to results/environment.txt; this is a
current snapshot, not an attestation of the environment at experiment time.
Verification: 18 correctness tests; temporary synthetic plotting/comparison
checks; portable ZIP extraction correctness check. No Banknote training or
saved experiment performance analysis was performed. Existing CSVs, models,
architecture selection, and experiment figures were preserved.
Visual suggestions in chat are guidance for students to create/select and explain
their own figures, not report prose or interpretation.

## 2026-11-26 — Claude (Anthropic) submission tidy-up
Prompt summary: check the code against the brief and prepare the submission zip.
Changes: comments and docstrings only (runner.py, smoke.py, make_tables_plots.py,
test_core.py, test_resume.py); README.md rewritten as a run guide; results/README.md
reworded. No functional code, experiment results or report text was changed.
