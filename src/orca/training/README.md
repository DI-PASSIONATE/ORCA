## Training

This folder contains the machine-learning side of ORCA: turning simulation results
into a trained surrogate model.

### Structure

Four objects, four concerns. Geometries own physics only; nothing about the
architecture lives in a geometry class.

| Module | Owns |
| --- | --- |
| `codecs.py` | `OutputCodec` — how a `skrf.Network` becomes a target tensor and back |
| `spec.py` | `IOSpec` / `FrequencyMode` — the shape contract between dataset and model |
| `datasets/` | Sample assembly from simulation results, normalization |
| `basis_expansion.py` | `BasisExpansion` — optional fixed expansions of the inputs, shared by all architectures |
| `models/` | `OrcaModel` — architecture, hyperparameter space, loss, guarantees |
| `trainer.py` | `Trainer` / `TrainingConfig` — optimizer, schedule, early stopping, history |
| `tuner.py` | `HyperparameterTuner` — optuna study with k-fold cross-validation |
| `losses.py` | Loss modules (`ComplexMSELoss`, `MSEPlusLogCoshLoss`) |

The model owns *what* is fitted (architecture and loss); the trainer owns *how* it is
fitted. `TrainingConfig` holds the hyperparameters the trainer owns, and
`TrainingConfig.from_hyperparameters()` picks them out of a combined dict, ignoring
architecture keys:

```python
trainer = Trainer(config=TrainingConfig(epochs=15, batch_size=256, learning_rate=5e-4))
result = trainer.fit(model, train_dataset, val_dataset)
result.best_loss, result.history, result.stopped_early
```

The model is chosen at the pipeline level, and so is the basis expansion:

```python
orca.ModelTrainer(model=orca.OrcaMLP, hyperparameters=...)          # or model="mlp"
orca.ModelTrainer(model="mlp", basis="chebyshev")                 # with a basis expansion
```

### Basis expansions

A `BasisExpansion` widens the input tensor before a model's first layer by appending
fixed basis functions of it. It is not tied to one architecture: any `OrcaModel`
takes one, so an expansion is written once and reused everywhere. Basis expansions
see **normalized** inputs in `(batch, n_inputs)` layout, and default to
`IdentityBasis` — expansion is opt-in.

| Basis | Adds |
| --- | --- |
| `identity` | nothing (default) |
| `chebyshev` | `degree` Chebyshev polynomials of one column, by default `frequency` |

`ChebyshevBasis` exists because per-point models take frequency as a single raw
column, so the whole frequency response has to be learned through one scalar; a
basis expansion of that column lets the network represent sharp, geometry-dependent
frequency structure without brute-forcing width. The mapped value is clamped to
[-1, 1], so a frequency beyond the training range saturates the basis instead of
diverging; the raw column is still passed through.

Because the basis expansion lives inside the model, it is tuned with it
(`basis_degree` joins the optuna study) and exported with it — `torch.onnx.export` traces it like
any other layer, so the ONNX input names are unchanged and nothing has to be
reproduced at inference time.

### Adding a basis expansion

```python
@register_basis("my_basis")
class MyBasis(BasisExpansion):
    def expanded_dim(self, input_dim: int) -> int: ...
    def forward(self, x): ...           # (batch, input_dim) -> (batch, expanded_dim)

    @classmethod
    def from_spec(cls, spec, hyperparameters): ...

    @staticmethod
    def hyperparameter_search_space():
        # merged into the study alongside the trainer's and the model's
        ...
```

### Adding a model

Subclass `OrcaModel`, size it from the `IOSpec` rather than from hardcoded
dimensions, and register it:

```python
@register_model("my_model")
class MyModel(OrcaModel):
    frequency_mode = FrequencyMode.PER_POINT
    guarantees = PhysicsGuarantees(reciprocal=True)

    @classmethod
    def from_spec(cls, spec, hyperparameters, basis=None): ...

    @staticmethod
    def hyperparameter_search_space():
        # architecture only - learning rate/batch size/epochs belong to the trainer
        ...
```

To accept any basis expansion, size the first layer from `self.expanded_dim` rather than
`spec.input_dim`, and start `forward` with `x = self.basis(x)`.

`guarantees` records what the architecture enforces *by construction*; it is written
into the exported ONNX metadata so downstream consumers know what they are holding.
`frequency_mode` is checked against the dataset when the trainer wires them together.