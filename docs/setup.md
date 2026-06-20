## Installation

## Requirements

- Python 3.11+
- Palace EM simulation software

Optional:

- COBRA installed/importable if you use ORCA surrogate outputs in circuit optimization

!!! note
	ORCA supports Python 3.11 to 3.14.

## Clone Repository

```bash
git clone https://github.com/DI-PASSIONATE/ORCA
cd ORCA
```

## Install ORCA

=== "Option A: uv (recommended)"

	```bash
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.13
	uv venv --python 3.13
	source .venv/bin/activate
	uv pip install -e .
	```

=== "Option B: venv + pip"

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip
	pip install -e .
	```

## Verify Setup

Run the following check in your activated environment:

```bash
orca
```

Expected behavior: ORCA GUI starts.

To verify the script workflow:

```bash
python examples/main.py
```

!!! warning
	`examples/main.py` requires a working Palace installation and a valid geometry configuration.

## External Tool Notes

### Palace

- ORCA uses Palace as the EM simulation backend.
- Install Palace by following [the official Palace installation instructions](https://awslabs.github.io/palace/stable/install/index.html).
- Palace can be run directly or via an Apptainer/Singularity container.

### COBRA (Optional)

- COBRA consumes ONNX surrogate models exported by ORCA for circuit-level optimization.
- See [COBRA documentation](https://di-passionate.github.io/COBRA/) for details.

## Next Steps

- Continue with **Getting Started -> Quickstart** for a first run.
- Use **User Guide -> Custom Classes** to bring your own geometry.