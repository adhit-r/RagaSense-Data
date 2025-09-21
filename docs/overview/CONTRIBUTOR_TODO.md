## Contributor TODO / Issues (Data Quality & Tooling)

Prioritized tasks derived from schemas, validators, processing scripts, and recent logs under `logs/`.

### P0 — Critical
1. Validator config portability
   - Replace hardcoded base paths (`/Users/adhi/axonome/RagaSense-Data`) with repo-relative default and `--base-path`/`RAGASENSE_BASE_PATH`.
   - Make W&B and GPU optional via flags/env; default off in CI.
   - Output reports to `logs/` under the chosen base path.

2. Ingestion reliability (from logs)
   - Fix invalid GitHub URLs for ramanarunachalam sources (404s in logs).
   - Implement Kaggle auth fallback (detect `KAGGLE_CONFIG_DIR` or `KAGGLE_USERNAME`/`KAGGLE_KEY`).
   - Add graceful no-op stubs for unimplemented downloaders with actionable errors.

3. YouTube URL/ID normalization
   - Normalize any `https://www.youtube.com/watch?v=...` and variants to canonical IDs.
   - Add a validator rule to flag malformed IDs and presence of extra query params (e.g., `&start=...&end=...`).
   - Backfill normalization across archived datasets where feasible.

### P1 — High
4. Schema tests and CI
   - Add sample JSONs for `carnatic` and `hindustani` under `data/samples/`.
   - Write pytest-based schema tests loading `schemas/metadata-schema.json`.
   - GitHub Actions: run tests, JSON linting, and validation dry-run.

5. Cleaner integration
   - Integrate `clean_and_deduplicate_ragas.py` with Make/CLI target to produce `data/cleaned_ragasense_dataset/*` artifacts.
   - Save a summary to `logs/raga_cleaning_report_*.json`.

6. Pre-commit hooks
   - Add `pre-commit` config: `jsonlint`, `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`, `check-merge-conflict`.
   - Optional: a local hook to validate JSON against schema for files under `data/**.json`.

### P2 — Medium
7. Data directory structure docs
   - Clarify `data/` active vs `archive/` historical datasets and how to reference them.
   - Document where logs, reports, and exports are written.

8. Cross-tradition mapping validation
   - Expand mapping schema tests and add consistency checks for many-to-one mappings.

9. Website data ingestion
   - Ensure `website/src/utils/data.ts` uses normalized IDs/paths; add type checks.

### P3 — Nice-to-have
10. W&B opt-in recipes
    - Provide example env/flags and a short guide under `docs/processing/`.

11. GPU acceleration toggles
    - Centralize GPU toggles, detect CUDA availability, and fallback robustly.

12. Dedup heuristics
    - Extend duplicate detection to handle transliteration variants and diacritics.

---

### How to Pick Up a Task
- Comment on the issue and wait for assignment.
- For P0/P1: write a short plan in the PR description and include test/verification steps.

### Definition of Done
- Tests and validators pass locally and in CI.
- No new schema or linter errors.
- Documentation updated where relevant.

