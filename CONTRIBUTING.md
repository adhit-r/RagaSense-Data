## Contributing to RagaSense-Data

Thank you for your interest in improving RagaSense-Data. This repo contains datasets, schemas, processing scripts, validators, and website artifacts. We welcome contributions focused on data quality, tooling, and documentation.

### Getting Started
- Clone the repo and create a new branch for your change.
- Use Python 3.10+; install dependencies: `pip install -r requirements.txt`.
- Optional: Node 18+ for `website/` workspace tasks.

### Areas to Contribute
- Data quality fixes (schema compliance, completeness, normalization)
- Ingestion sources (URLs, authentication flows, parsers)
- Validation tooling (`tools/validation`) and tests
- Cleaning/deduplication pipeline integration (`scripts/data_processing`)
- Documentation (READMEs under `docs/`, contributor guides)

### Workflow
1. Create an issue or pick one from the TODO/Issues list.
2. Reference the issue ID in your branch name and commit messages.
3. Add or update tests/docs as needed.
4. Open a PR; include a brief description and reproduction/verification steps.

### Code Quality
- Match existing formatting; avoid unrelated refactors.
- Prefer clarity over cleverness; handle edge cases.
- Keep functions small and well-named; write high-verbosity code.
- Run linters and validators before pushing.

### Data Safety
- Do not commit large raw datasets unless required and licensed.
- Do not include secrets (e.g., Kaggle API tokens). Use env vars.

### Contact
Questions? Open a discussion or issue.

