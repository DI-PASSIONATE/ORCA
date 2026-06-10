# Contributing to ORCA

Thank you for your interest in contributing to ORCA! There are several ways to get involved — from reporting bugs and improving documentation, to submitting code changes, or helping grow the community surrogate model library on Hugging Face.

## Table of Contents

- [Reporting Issues](#reporting-issues)
- [Contributing Code](#contributing-code)
  - [Setting Up a Development Environment](#setting-up-a-development-environment)
  - [Branching and Commit Style](#branching-and-commit-style)
  - [Opening a Pull Request](#opening-a-pull-request)
  - [Code Style](#code-style)
- [Contributing Surrogate Models](#contributing-surrogate-models)
  - [Step 1 — Train a Surrogate with ORCA](#step-1--train-a-surrogate-with-orca)
  - [Step 2 — Create a Hugging Face Repository](#step-2--create-a-hugging-face-repository)
  - [Step 3 — Add a Model Card](#step-3--add-a-model-card)
  - [Step 4 — Upload the Model Files](#step-4--upload-the-model-files)
  - [Step 5 — Verify the Repository](#step-5--verify-the-repository)
- [Community Guidelines](#community-guidelines)

---

## Reporting Issues

If you encounter a bug, unexpected behavior, or have a feature request:

1. **Search existing issues** at [github.com/DI-PASSIONATE/ORCA/issues](https://github.com/DI-PASSIONATE/ORCA/issues) to avoid duplicates.
2. **Open a new issue** and fill in as much detail as possible:
   - A clear, descriptive title.
   - Steps to reproduce the problem.
   - Expected vs. actual behavior.
   - Your Python version, OS, and ORCA version (`pip show orca`).
   - Any relevant error messages or tracebacks.
3. For feature requests, describe the use case and why the feature would be valuable.

---

## Contributing Code

### Setting Up a Development Environment

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/ORCA
   cd ORCA
   ```

2. Create and activate a virtual environment:

   ```bash
   # Using uv (recommended)
   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install -e ".[dev]"

   # Or using standard venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev]"
   ```

3. Add the upstream remote so you can stay up to date:

   ```bash
   git remote add upstream https://github.com/DI-PASSIONATE/ORCA
   git fetch upstream
   ```

### Branching and Commit Style

- Create a new branch for each contribution:

  ```bash
  git checkout -b feature/my-new-feature
  # or
  git checkout -b fix/issue-123-short-description
  ```

- Write clear, concise commit messages in the imperative mood:

  ```
  Add support for rectangular spiral geometry
  Fix normalization bug in ModelTrainer for single-port devices
  ```

- Keep commits focused — one logical change per commit.

### Opening a Pull Request

1. Push your branch to your fork:

   ```bash
   git push origin feature/my-new-feature
   ```

2. Open a pull request against the `main` branch of `DI-PASSIONATE/ORCA`.

3. In the PR description:
   - Summarize what the change does and why.
   - Reference any related issues (e.g. `Closes #42`).
   - Describe how the change was tested.

4. A maintainer will review your PR. Please respond to review comments promptly and push any requested changes to the same branch — the PR will update automatically.

### Code Style

- ORCA follows [PEP 8](https://peps.python.org/pep-0008/). Run `ruff check .` or `flake8` before submitting.
- Type annotations are encouraged for new public functions and methods.
- Keep new dependencies minimal. If you need a new dependency, discuss it in the issue or PR first.

---

## Contributing Surrogate Models

One of the most impactful ways to contribute to ORCA is to run EM simulations for a passive component, train a surrogate model, and share it publicly on Hugging Face. Other users — and [COBRA](https://github.com/DI-PASSIONATE/COBRA) — can then use your model directly without re-running expensive EM simulations.

### Step 1 — Train a Surrogate with ORCA

Write a geometry class that extends `BaseGeometry` (see the [Custom Classes documentation](docs/custom_class.md) and the built-in `TransformerOcta` preset as a reference). Then run the full ORCA pipeline:

```python
import orca
from orca import ORCA
from your_geometry_module import YourGeometry

geometry = YourGeometry()

orca_instance = ORCA(
    [
        orca.GDSGenerator(num_samples=1000),
        orca.GDSConverter(),
        orca.PalaceSimulator(palace_executable="palace"),
        orca.ModelTrainer(),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

orca_instance.run(geometry=geometry, cpu_cores=16)
```

This produces:
- `<model_name>.onnx` — the portable surrogate model.
- `<model_name>.py` — your geometry class file.

### Step 2 — Create a Hugging Face Repository

1. Sign in to [huggingface.co](https://huggingface.co) (create a free account if needed).
2. Go to [huggingface.co/new](https://huggingface.co/new) and create a new model repository.
   - Choose a descriptive name (e.g. `ihp-sg13g2-transformer-octa`).
   - Set visibility to **Public**.
   - Note your repository ID, which takes the form `your-username/your-model-name`.
3. Click **Create model**.

### Step 3 — Add a Model Card

A Model Card is a structured README that describes your model. It makes your model discoverable by COBRA and other ORCA users.

1. In your new repository, click **Add Model Card**.
2. At the top of the file add the `orca-surrogate` tag so the model appears in community searches:

   ```markdown
   ---
   tags:
   - orca-surrogate
   ---
   ```

3. Fill in the rest of the card with a description of the component, the technology node, the parameter ranges covered, and any relevant simulation settings or caveats.

### Step 4 — Upload the Model Files

Install the Hugging Face Hub client if you haven't already:

```bash
pip install huggingface_hub
```

Log in with your Hugging Face credentials:

```bash
huggingface-cli login
```

Then upload both required files:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/your-model-name"  # replace with your repository ID

api.upload_file(
    path_or_fileobj="your_model_name.onnx",
    path_in_repo="your_model_name.onnx",
    repo_id=repo_id,
)
api.upload_file(
    path_or_fileobj="your_model_name.py",
    path_in_repo="your_model_name.py",
    repo_id=repo_id,
)
```

Alternatively, upload the files through the web interface: go to your repository → **Files** → **Add file → Upload files**.

| File | Description |
|------|-------------|
| `<model_name>.onnx` | The exported ONNX surrogate model produced by `OnnxExporter` |
| `<model_name>.py` | The Python geometry class (subclass of `BaseGeometry`) used to generate and train the model |

Both files must be present and share the same base name for COBRA to load the model correctly.

### Step 5 — Verify the Repository

Before announcing your model, confirm:

- [ ] The repository is **public**.
- [ ] The Model Card contains the `orca-surrogate` tag.
- [ ] Both `<model_name>.onnx` and `<model_name>.py` are present in the repository.
- [ ] The geometry class file is self-contained (all custom imports are available or documented).

Once published, your model will automatically be discoverable by COBRA and anyone searching for `orca-surrogate` models on Hugging Face. Consider also opening a [discussion or issue](https://github.com/DI-PASSIONATE/ORCA/issues) in the ORCA repository to announce your contribution so the community knows about it!

---

## Community Guidelines

- Be respectful and constructive in all interactions.
- Assume good faith from other contributors.
- If you are unsure whether a change is in scope, open an issue to discuss it before investing time in an implementation.
- For substantial changes or new pipeline stages, please discuss the design in an issue first.

We appreciate every contribution, large or small. Thank you for helping make ORCA better!
