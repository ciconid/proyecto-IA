# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University assignment (UNS, Inteligencia Artificial 2026): ternary classification of movies into
`Mala` / `Buena` / `Muy buena` using two models — a **decision tree** and an **artificial neural
network** — with Pandas, Scikit-learn and Keras. Full spec: `docs/2026 - IA - Proyecto Aprendizaje Automático.docx.pdf`
(in Spanish; work and notebook text are in Spanish).

No code exists yet; the deliverable is a Jupyter notebook (runnable in Google Colab).

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
