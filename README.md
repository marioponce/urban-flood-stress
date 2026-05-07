![CI](https://github.com/youruser/project_name/actions/workflows/ci.yml/badge.svg)
# Python-Project-Template

This repository is a **project template** for Python-based scientific or analytical work.  
It provides a standardized structure, development tools, and configuration to quickly start new projects.

Before using it, you must modify the configuration and metadata files according to your specific project.

---

## Required Modifications

Before starting development, update the following files:

- `pyproject.toml`
  - Change `project_name`
  - Update description
  - Update author name and email
  - Adjust dependencies if needed

- `environment.yml`
  - Adjust Python version if necessary
  - Modify project-specific dependencies

- `README.md`
  - Replace this template description with your project description

- `CITATION.cff`
  - Update title
  - Update version
  - Update release date
  - Update repository URL
  - Adjust authors and affiliations

- `src/project_name/`
  - Rename the package folder to match your project name

Failing to update these fields may result in incorrect metadata or citation information.

---

## Installation

### 1. Modify environment configuration

Edit the following files according to your project needs:

- `environment.yml` → adjust Python version and dependencies  
- `pyproject.toml` → update project name, description, authors, and dependencies  

Then create the environment:

```bash
conda env create -f environment.yml
conda activate project_env
```
---
### 2. Modify project metadata

Before installation, update:

* project_name in pyproject.toml
* the folder name inside src/
* this README.md
* author information and contact email

Then install in editable mode:
```bash
pip install -e .[dev]
```

If you need documentation dependencies:

```bash
pip install -e .[dev,docs]
```
---
### Running Tests
```bash
pytest
```
---
### Linting & Formatting

Check code quality:

```bash
ruff check .
```

Auto-fix issues:

```bash
ruff check --fix .
```

Format code:
```bash
ruff format .
```
---
### Pre-commit Hooks (Recommended)

To automatically run linting and formatting before each commit:

```bash
pip install -e .[dev]
pre-commit install
```

Now, Ruff will run automatically before every commit.

To run manually on all files:
```bash
pre-commit run --all-files
```
---
### Build Documentation (Sphinx)

From the docs/ directory:

```bash
make html
```

The built docs will be available in:

```bash
docs/_build/html/index.html
```
---
### Project Structure

```bash
project_name/
│
├── src/project_name/
├── tests/
├── docs/
├── pyproject.toml
├── environment.yml
└── README.md
```
---
### License

MIT License.

---

### Versioning Strategy

This project follows a simplified Semantic Versioning scheme:

MAJOR.MINOR.PATCH

- **PATCH** (0.1.1 → 0.1.2):  
  Bug fixes, small refactors, documentation updates.  
  No breaking changes.

- **MINOR** (0.1.0 → 0.2.0):  
  New functionality added in a backward-compatible way.

- **MAJOR** (0.x.x → 1.0.0 or 1.0.0 → 2.0.0):  
  Breaking changes that modify public APIs or expected behavior.

#### Practical Rule

If someone updates the package and their existing code continues to work → use PATCH or MINOR.  
If updating the package breaks existing code → use MAJOR.

For early-stage research projects, versions typically remain in the 0.x.x range.  
A stable release used in publications or reports should be marked as 1.0.0.