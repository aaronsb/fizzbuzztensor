# Contributing

PRs are welcome for new representations, visualizations, generalizations, or
sections of the paper.

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

`uv` installs the pinned dependencies from `uv.lock`. If you prefer plain pip:
`pip install -e ".[dev]"`.

## Ground rules

- Every representation must pass `tests/test_fizzbuzz.py`, which compares it
  against the naive reference implementation on a range of inputs.
- If you add a representation, add it to the test file and to the comparison
  table in the paper.
- If you change a visualization, regenerate its figure and commit the PNG under
  `docs/images/`.
- Open a PR against `main`. CI runs lint, the test matrix, and a headless
  render of every figure.
