## Setup

Required tools:

- Python 3.11 or higher
- Python package manager (Recommended: `uv`)
- virtual environment tool (e.g., venv or conda)
- Palace EM simulation software (open-source, available at https://github.com/awslabs/palace)

### Installation Steps

- Clone this repository

```bash
git clone https://github.com/DavidL-11/ORCA && cd ORCA
```

- Install UV (https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Download Python 3.13

```bash
uv python install 3.13
```

- Create and activate a virtual environment

```bash
uv venv --python 3.13
```

- Install ORCA

```bash
uv pip install -e .
```

Install Palace on your system by following [the Palace installation instructions](https://awslabs.github.io/palace/stable/install/index.html).