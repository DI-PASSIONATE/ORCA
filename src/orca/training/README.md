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
| `guarantees.py` | `PhysicsGuarantees` — properties a model or codec enforces by construction |
| `models/` | `OrcaModel` — architecture, hyperparameter space, loss, guarantees |
| `trainer.py` | `Trainer` / `TrainingConfig` — optimizer, schedule, early stopping, history |
| `tuner.py` | `HyperparameterTuner` — optuna study with k-fold cross-validation over geometries, or one fixed hold-out (`n_fold_cv=1`) |
| `losses.py` | Loss modules (`ComplexMSELoss`, `MSEPlusLogCoshLoss`, `SParameterLoss`), self-resonance detection and sample weights |

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

### Output codecs

An `OutputCodec` decides *what* is regressed. It is the one place that knows how a
`skrf.Network` becomes a target tensor and how model outputs become an S-matrix
again, so datasets, trainer, exporter and predictors are all representation-agnostic.
The codec is chosen on the dataset, in the geometry preset:

```python
GeoToSParamDatasetSingleFrequency(
    codec=UpperTriangleReImCodec(n_ports=6),
    ...
)
```

| Codec | Values per frequency point | Notes |
| --- | --- | --- |
| `FlatReImCodec` | `2·N²` (72 for 6 ports) | Every entry independently; ORCA's historical layout |
| `UpperTriangleReImCodec` | `N·(N+1)` (42 for 6 ports) | Upper triangle only, mirrored on decode |

`UpperTriangleReImCodec` exploits reciprocity: a passive reciprocal structure
satisfies `S = Sᵀ`, so the lower triangle is redundant — 21 unique complex entries
for 6 ports, not 36. Predicting only those roughly halves the output dimension and
makes symmetry structural rather than something the network has to learn and a
downstream optimizer can exploit. On encode the two measured halves are averaged
(the projection onto the reciprocal subspace) instead of one being thrown away.
Use `FlatReImCodec` for anything non-reciprocal.

Changing the codec changes the ONNX output signature (`S12_real`, ... in codec
order, no lower-triangle entries), so a consumer reading outputs positionally has
to be re-pointed. Codec and model both declare `PhysicsGuarantees`; the exporter
writes their union into the ONNX metadata, so a model exported with the
upper-triangle codec is marked `reciprocal` whatever the architecture is.

### Loss terms and sample weights

A model trains with its `default_loss()` (L1 on the normalized outputs for the MLP).
`SParameterLoss` wraps that data loss with terms that judge the prediction as a circuit,
and `ModelTrainer` builds it from three settings:

| Setting | Adds |
| --- | --- |
| `admittance_weight` | `admittance_error`: relative L1 error of Y = (I + S)⁻¹(I − S), of all of Y and of its real part, which track L and Q |
| `passivity_weight` | `passivity_violation`: how far σ_max of the predicted S exceeds max(1, σ_max of the simulated S) |
| `above_srf_weight` | per-sample weights from `srf_sample_weights`: points above each geometry's first self-resonance count this much relative to those below |

The terms are computed per sample on the denormalized S-matrices, which the codec's
`decode_tensor` rebuilds, so the data loss has to be elementwise (a torch loss with a
`reduction` attribute). Sample weights live on the dataset (`set_sample_weights`), are
appended to its `tensors` and batched like the inputs, also through the tuner's `Subset`
folds; the trainer passes each batch's weights to the loss as a third argument. They are
normalized to average 1, so the epoch loss stays on the scale of the data loss.

`first_self_resonance` needs no knowledge of the ports: it counts the negative eigenvalues
of the susceptance matrix Im(Y), one per inductive mode, and returns where their number
first drops. For InductorOcta this is the differential self-resonance.

### Adding an output codec

```python
class MyCodec(OutputCodec):
    guarantees = PhysicsGuarantees(reciprocal=True)   # only if violations are impossible

    @property
    def output_dim(self) -> int: ...
    @property
    def output_names(self) -> list[str]: ...          # ONNX output names, in encode order

    def encode(self, ntwk): ...                       # skrf.Network -> (n_freq, output_dim)
    def decode(self, raw): ...                        # (n_freq, output_dim) -> (n_freq, N, N) complex
    def decode_tensor(self, raw): ...                 # the same in torch, differentiable; optional,
                                                      # needed by the admittance and passivity losses
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

    @staticmethod
    def regularization_search_space():
        # optional, e.g. dropout; only searched with ModelTrainer(regularization=True)
        ...
```

To accept any basis expansion, size the first layer from `self.expanded_dim` rather than
`spec.input_dim`, and start `forward` with `x = self.basis(x)`.

`guarantees` records what the architecture enforces *by construction*; it is written
into the exported ONNX metadata so downstream consumers know what they are holding.
`frequency_mode` is checked against the dataset when the trainer wires them together.