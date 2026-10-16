# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University assignment (UNS, Inteligencia Artificial 2026): ternary classification of movies into
`Mala` / `Buena` / `Muy buena` using two models — a **decision tree** and an **artificial neural
network** — with Pandas, Scikit-learn and Keras. Full spec: `docs/2026 - IA - Proyecto Aprendizaje Automático.docx.pdf`
(in Spanish; work and notebook text are in Spanish).

The deliverable is a single Jupyter notebook, `notebooks/proyecto.ipynb` (must stay Colab-compatible).
Development plan and phases: `PLAN.md`. Experiment log: `docs/experimentos.md`.

**Working mode:** the student writes the notebook code to learn it (they must defend it orally).
Explain concepts, give hints and review their code. Claude writes only the boilerplate: section
headers, imports, data loading/saving and trivial inspection (`shape`, `head`, `info`). The student
writes everything involving a decision or analysis: EDA findings, preprocessing, models, experiments.

**Commits:** the student commits. After any change to the repo (yours or theirs), end your reply with
a suggested commit message in Spanish, imperative mood (e.g. `Agregar análisis exploratorio`).
Show only the message, not the git commands: a line with just `commit-message:`, a blank line, then the
message alone on its own line (the blank line keeps markdown from joining them), with no backticks or other text, so a triple-click copies only the message. Don't run `git commit` unless they ask.

## Commands

```bash
source .venv/bin/activate                     # venv (Python 3.12) with requirements.txt installed
jupyter notebook notebooks/                   # or open in VS Code with the .venv kernel
jupyter nbconvert --to notebook --execute notebooks/proyecto.ipynb --output /tmp/run.ipynb  # run end to end
```

## Deliverables (zip uploaded to Moodle; deadline 22/10/2026 23:59, resubmission 05/11/2026)
- One `.ipynb` containing data preprocessing, both models and their evaluation. Should be well
  organized with markdown cells as section headers; code must be clear (comments optional).
- One `.csv`: a copy of the test file plus two extra columns — decision-tree predictions and ANN
  predictions.
- Presentation slides as `.pdf` (10-minute talk).

## Grading priorities
- Preprocessing is graded as part of the solution; metric scores matter but are not the main goal.
- Keep a record of experimentation: initial hyperparameters, how and in what order they were varied,
  and results — the presentation must describe this evolution.

## Data
- `data/movies_train.csv` — labeled training data (target column: `class`).
- `data/movies_test_actualizado.csv` — test instances to predict. Note it also contains a `class`
  column; don't train on it.
- Columns: movie info (`movieId`, `title` with year in parentheses, `genre`, `belong_to_collection`,
  `budget`, `popularity`, `revenue`, `runtime`, `production_company_name`/`_country`), three lead
  performers (`name_pN`, `popularity_pN`, `number_movies_pN`, `popularity_movies_pN` for N=1..3),
  and `director_name`, `director_popularity`.
- Known data quirks: `budget`/`revenue` use `0` (and sometimes empty) for unknown values; missing
  values appear in `runtime`, `revenue`, `production_company_country`, `director_popularity`, etc.;
  text fields contain non-ASCII characters (read as UTF-8).
