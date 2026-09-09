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
| `datasets/` | Sample assembly from simulation results, features, normalization |
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

The model is chosen at the pipeline level:

```python
orca.ModelTrainer(model=orca.OrcaMLP, hyperparameters=...)   # or model="mlp"
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
    def from_spec(cls, spec, hyperparameters): ...

    @staticmethod
    def hyperparameter_search_space():
        # architecture only - learning rate/batch size/epochs belong to the trainer
        ...
```

`guarantees` records what the architecture enforces *by construction*; it is written
into the exported ONNX metadata so downstream consumers know what they are holding.
`frequency_mode` is checked against the dataset when the trainer wires them together.