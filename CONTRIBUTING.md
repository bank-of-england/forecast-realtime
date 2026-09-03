# Contributor Guide
## Initial Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/bank-of-england/forecast-realtime.git
cd forecast-realtime
```

2. **Set up the development environment**

Create and activate a fresh environment, then install the package with the
full contributor dependency set:
```bash
pip install -e ".[dev]"
```
The `dev` extra pulls in the docs, notebook, and model dependencies as well,
so this is the only install command you need.

3. **Install pre-commit hooks**
```bash
pre-commit install
```

Pre-commit runs the following checks when you commit changes:

- Ruff's linter, with automatic fixes where possible
- Ruff's formatter
- API documentation generation from the public exports in `src/forecast_realtime`
- Notebook Markdown freshness checks
- NumPy-style docstring validation with `pydoclint`
- The full pytest suite

The generated API manifest is written to `docs/api.md`. If a source module's
public API changes, the hook updates that file so it can be included in the
same commit. To run every hook across the repository without creating a
commit, use:

```bash
pre-commit run --all-files
```

Pre-commit does not build the documentation site or distribution package; run
the commands in the [Documentation](#documentation) and [Code Style](#code-style)
sections when you need to check those artefacts.

4. **Verify the installation**
```bash
pytest
```

## Development Workflow

### Branch Strategy

- **`main`**: Production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/issue-description`
- **Documentation**: `docs/topic-name`

## Protected Branches and Pull Requests

All contributions must be submitted through a pull request. The `main` branch
is protected, so contributors cannot push changes directly to it. Create a
branch from `main`, commit and push your changes there, then open a pull request
targeting `main`. The required check is the `package-quality` workflow, which
must pass before the pull request can merge.
Automated dependency and release changes follow the same pull-request process.

### Creating a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### Keeping Your Branch Updated

```bash
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main  # Merge main instead when necessary.
```

### Commit your changes

Use [Conventional Commit](https://www.conventionalcommits.org/) subjects
(`fix:`, `feat:`, `deps:`, `docs:`, `chore:`, ...); Release Please builds the
changelog and the next version from them.

```bash
git add .
git commit -m "fix: describe your change"
git push  # Or specify the branch explicitly.
```

## Code Standards

### Code Style

We use **Ruff** for formatting and linting:

```bash
# Format the code.
ruff format .

# Check for lint issues.
ruff check .

# Fix issues that Ruff can resolve.
ruff check . --fix

# Check formatting without changing files.
ruff format --check .

# Build the package.
python -m build

# Build the documentation.
zensical build
```

## Documentation

Zensical builds the documentation site. `mkdocstrings` renders the public
objects listed in each package's `__all__` declaration. The generated
`docs/api.md` manifest connects those objects to the API reference. Edit the
source docstrings and public exports, not the generated directives.

The documentation dependencies are already installed with the `dev` extra
(`pip install -e ".[dev]"`); `.[docs,notebooks]` installs just those if you
need a docs-only environment.

Regenerate the API manifest explicitly when needed:

```bash
python scripts/generate_api_docs.py
```

Marimo apps belong in `notebooks/`. Export them to committed Markdown pages
under `docs/notebooks/`, or check that existing exports are current:

```bash
python docs/convert_notebooks.py
python docs/convert_notebooks.py --check
```

Build the complete documentation site locally, including strict validation:

```bash
zensical build --clean --strict
```

Continuous integration regenerates `docs/api.md` and fails when the committed
manifest is stale.

### Naming Conventions

- **Variables**: `snake_case`
- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions/methods**: `_leading_underscore`

## Submitting Changes

### Before Submitting

1. **Open an issue** to discuss the bug or feature.

2. **Use the issue number in the branch name; for example, `fix/1-prior`.**

3. **Make your changes**

4. **Add a test for the change.**


5. **Format, document, and test the code.**
```bash
ruff format
ruff check .
pytest
```

6. **Commit and push the changes** with a Conventional Commit subject.
```bash
git add .
git commit -m "fix: describe your change (#1)"
git push origin fix/1-prior
```

7. **Submit a pull request.**

## Creating a Release (for maintainers)

Releases are automated with [Release Please](https://github.com/googleapis/release-please).
`release-please.yml` watches `main` for [Conventional Commits](https://www.conventionalcommits.org/)
and opens or updates a release pull request carrying the next version and the
generated `CHANGELOG.md` entries. Only recognised subject types
(`fix:`, `feat:`, `deps:`, ...) are picked up; an untyped subject is ignored.

`release-please-config.json` sets an `always-bump-patch` strategy, so every
release is a patch bump: `fix:`, `feat:`, `deps:`, and breaking commits all
take `0.5.6` to `0.5.7`. Commit types still organise the changelog but do not
change the version bump. A `Release-As: 0.6.0` footer on a typed commit is an
exact one-time override; it is not needed for normal changes.

`release-please.yml` enables auto-merge on the release pull request, so GitHub
merges it automatically once the required `package-quality` check and branch
protection pass. The `ecosystem` gate is deliberately skipped on Release Please
pull requests (the version bump makes the candidate wheel run ahead of the
pinned ecosystem), so it must not be a required check. Because of that, every
merge to `main` with a typed commit produces a release; keep in-progress work on
your integration branch until it is ready to ship.

When the release pull request merges, Release Please creates the `v<version>`
tag and the GitHub Release. The published release then starts:

- `publish-pypi.yml` — builds the distribution and publishes it to PyPI, then
  triggers `update-ecosystem.yml` to re-pin `forecast_realtime` in `opera-eco`.
- `deploy-docs.yml` — builds the documentation site and deploys it to GitHub Pages.

Both also support manual dispatch against an existing tag.

### One-time setup

- Add a `RELEASE_PLEASE_TOKEN` repository secret: a token that can write
  contents, issues, pull requests, tags, and releases. A plain `GITHUB_TOKEN`
  will not do, because the release it creates must be able to trigger the
  downstream publication and documentation workflows.
- Enable **Allow auto-merge** in the repository settings.
- In the `github-pages` environment, keep the `main` deployment branch rule
  and add a `v*` tag rule. Release-triggered documentation runs use the release
  tag and are rejected before deployment when that tag is not allowed.
- Require the `package-quality` check on `main` (not `ecosystem` — it is
  skipped on release pull requests, and a skipped required check blocks
  auto-merge), and make sure required human reviews do not block the
  automation pull requests.