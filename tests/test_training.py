"""Training, tuning and testing on a small synthetic 2-port dataset (no Palace needed)."""

import itertools
import json
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
skrf = pytest.importorskip("skrf")

from orca import ORCA
from orca.geometry.base_geometry import BaseGeometry
from orca.geometry.input_parameters import InputParameterIterator, RangeParameter
from orca.pipeline import training_stage
from orca.pipeline.context import PipelineContext
from orca.pipeline.test_model_stage import ModelTester
from orca.pipeline.training_stage import ModelTrainer, order_parameter_columns
from orca.training.codecs import FlatReImCodec
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.datasets.geo_to_s_param_single_f import (
    GeoToSParamDatasetSingleFrequency,
)
from orca.training.models.mlp import OrcaMLP
from orca.training.normalize import MinMaxNormalizer, StandardNormalizer
from orca.training.trainer import (
    DEFAULT_BATCH_SIZES,
    LearningRateSchedule,
    Trainer,
    TrainingConfig,
    make_batches,
)
from orca.training.tuner import HyperparameterTuner, group_kfold_indices

FREQUENCIES = np.linspace(1e9, 10e9, 10)
A_VALUES = [1.0, 2.0, 3.0, 4.0]
B_VALUES = [0.5, 1.0, 1.5]
TINY_MLP = {"epochs": 2, "batch_size": 16, "num_layers": 1, "hidden_size": 8}


def _iterator(*extra: RangeParameter) -> InputParameterIterator:
    return InputParameterIterator(
        RangeParameter("a", A_VALUES[0], A_VALUES[-1], step=1.0),
        RangeParameter("b", B_VALUES[0], B_VALUES[-1], step=0.5),
        *extra,
        frequency=[FREQUENCIES[0], FREQUENCIES[-1]],
    )


def _dataset(iterator: InputParameterIterator) -> GeoToSParamDatasetSingleFrequency:
    return GeoToSParamDatasetSingleFrequency(
        codec=FlatReImCodec(n_ports=2),
        input_normalizer=MinMaxNormalizer(iterator),
        output_normalizer=StandardNormalizer(),
    )


@dataclass
class ToyGeometry(BaseGeometry):
    name: str = "toy"
    stackup_xml: str = ""
    simconfig_filename: str = ""
    input_parameter_iterator: InputParameterIterator = field(default_factory=_iterator)

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict) -> str:
        raise NotImplementedError

    def create_dataset(self) -> GeoToSParamDatasetSingleFrequency:
        return _dataset(self.input_parameter_iterator)


@pytest.fixture
def result_dir(tmp_path) -> str:
    """One smooth, reciprocal 2-port response per (a, b) pair, plus the parameter table."""
    rows = []
    for i, (a, b) in enumerate((a, b) for a in A_VALUES for b in B_VALUES):
        phase = np.exp(-1j * FREQUENCIES / 1e10 * a)
        s21 = 0.9 * phase / (1 + 0.1 * b)
        s11 = 0.1 * b * phase**2
        s = np.empty((len(FREQUENCIES), 2, 2), dtype=complex)
        s[:, 0, 0] = s[:, 1, 1] = s11
        s[:, 0, 1] = s[:, 1, 0] = s21
        ntwk = skrf.Network(frequency=skrf.Frequency.from_f(FREQUENCIES, unit="hz"), s=s)
        ntwk.write_touchstone(filename=f"toy_{i}", dir=str(tmp_path))
        rows.append({"name": f"toy_{i}.s2p", "a": a, "b": b})
    pd.DataFrame(rows).to_csv(tmp_path / "toy.csv", index=False)
    return str(tmp_path)


def _table(result_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(result_dir, "toy.csv"))


def _loaded(result_dir: str) -> BaseDataset:
    return _dataset(_iterator()).new_split(result_dir, _table(result_dir), fit_normalizers=True)


def test_dataset_records_one_group_per_sample(result_dir):
    dataset = _loaded(result_dir)

    assert len(dataset) == len(A_VALUES) * len(B_VALUES) * len(FREQUENCIES)
    assert len(dataset.sample_groups) == len(dataset)
    assert dataset.sample_groups.count("toy_0.s2p") == len(FREQUENCIES)
    inputs, targets = dataset.tensors
    assert inputs.shape == (len(dataset), 3)
    assert targets.shape == (len(dataset), 8)


def test_group_kfold_never_splits_a_geometry(result_dir):
    groups = _loaded(result_dir).sample_groups

    folds = group_kfold_indices(groups, n_splits=3, seed=0)

    seen_in_validation = set()
    for train, val in folds:
        train_groups = {groups[i] for i in train}
        val_groups = {groups[i] for i in val}
        assert not train_groups & val_groups
        assert len(train) + len(val) == len(groups)
        seen_in_validation |= val_groups
    assert seen_in_validation == set(groups)


def test_group_kfold_needs_a_geometry_per_fold():
    with pytest.raises(ValueError, match="at least 5 geometries"):
        group_kfold_indices(["x.s2p", "x.s2p", "y.s2p"], n_splits=5, seed=0)


def test_normalizer_column_order_is_checked(result_dir):
    swapped = _table(result_dir)[["name", "b", "a"]]

    with pytest.raises(ValueError, match="input normalizer expects"):
        _dataset(_iterator()).new_split(result_dir, swapped, fit_normalizers=True)

    reordered = order_parameter_columns(swapped, ToyGeometry())
    assert list(reordered.columns) == ["name", "a", "b"]


def test_parameter_table_with_extra_columns_is_rejected(result_dir):
    table = _table(result_dir).assign(note="x")

    with pytest.raises(ValueError, match="unexpected \\['note'\\]"):
        order_parameter_columns(table, ToyGeometry())


def test_fixed_parameter_normalizes_to_finite_values(result_dir):
    table = _table(result_dir).assign(c=2.0)[["name", "a", "b", "c"]]
    iterator = _iterator(RangeParameter("c", 2.0, 2.0))

    dataset = _dataset(iterator).new_split(result_dir, table, fit_normalizers=True)

    assert torch.isfinite(dataset.inputs).all()
    assert torch.all(dataset.inputs[:, 2] == 0)


def test_losses_are_weighted_by_batch_size(result_dir):
    dataset = _loaded(result_dir)
    model = OrcaMLP.from_spec(dataset.io_spec, {"num_layers": 1, "hidden_size": 8})
    trainer = Trainer(TrainingConfig(device=dataset.device), verbose=False)

    # 120 samples in batches of 50: the last batch holds 20
    batched = trainer.evaluate(model, dataset, criterion=torch.nn.L1Loss(), batch_size=50)

    with torch.no_grad():
        inputs, targets = dataset.tensors
        whole = torch.nn.functional.l1_loss(model.to(dataset.device)(inputs), targets).item()
    assert batched == pytest.approx(whole, rel=1e-5)


def test_subsets_are_batched_from_stacked_tensors(result_dir):
    dataset = _loaded(result_dir)
    subset = torch.utils.data.Subset(dataset, [5, 2, 7])

    (x, y), *_ = list(make_batches(subset, batch_size=10, shuffle=False))

    assert torch.equal(x, dataset.inputs[[5, 2, 7]])
    assert torch.equal(y, dataset.targets[[5, 2, 7]])


class _BrokenMLP(OrcaMLP):
    def forward(self, x):
        raise TypeError("a bug in the model")


class _DivergingMLP(OrcaMLP):
    def forward(self, x):
        return super().forward(x) * math.nan


def test_tuner_scores_a_single_fold_on_the_holdout(result_dir):
    dataset = _loaded(result_dir)
    holdout = {"toy_0.s2p", "toy_5.s2p", "toy_11.s2p"}
    tuner = HyperparameterTuner(
        OrcaMLP, dataset, n_fold_cv=1, n_trials=1, max_epochs=2, holdout_groups=holdout
    )

    ((train, val),) = tuner.folds
    assert {dataset.sample_groups[i] for i in val} == holdout
    assert not {dataset.sample_groups[i] for i in train} & holdout
    assert len(train) + len(val) == len(dataset)

    tuner.tune()

    assert tuner.study is not None
    (trial,) = tuner.study.trials
    assert sorted(trial.intermediate_values) == [0, 1]


def test_holdout_tuning_needs_holdout_groups(result_dir):
    dataset = _loaded(result_dir)
    with pytest.raises(ValueError, match="pass its geometries as holdout_groups"):
        HyperparameterTuner(OrcaMLP, dataset, n_fold_cv=1)
    with pytest.raises(ValueError, match="at least 1"):
        HyperparameterTuner(OrcaMLP, dataset, n_fold_cv=0)


def test_stage_tunes_on_its_validation_split_with_one_fold(result_dir, tmp_path, monkeypatch):
    holdouts = []

    class RecordingTuner(HyperparameterTuner):
        def __init__(self, *args, **kwargs):
            holdouts.append(kwargs["holdout_groups"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(training_stage, "HyperparameterTuner", RecordingTuner)
    context = PipelineContext(
        geometry=ToyGeometry(), base_dir=str(tmp_path / "run"), result_dir_override=result_dir, seed=2
    )
    trainer = ModelTrainer(
        n_fold_cv=1, n_trials=1, tuning_max_epochs=1, max_epochs=1, batch_sizes=[16]
    )

    context = trainer.run(context)

    split = pd.read_csv(context.split_csv_path)
    assert holdouts == [set(split.loc[split["split"] == "val", "name"])]


def test_tuner_reraises_bugs_instead_of_pruning(result_dir):
    tuner = HyperparameterTuner(_BrokenMLP, _loaded(result_dir), n_fold_cv=2, n_trials=1)

    with pytest.raises(TypeError, match="a bug in the model"):
        tuner.tune()


def test_tuner_prunes_diverging_trials(result_dir):
    tuner = HyperparameterTuner(_DivergingMLP, _loaded(result_dir), n_fold_cv=2, n_trials=2)

    with pytest.raises(RuntimeError, match="None of the 2 tuning trials completed"):
        tuner.tune()
    assert tuner.study is not None
    assert all(t.state.name == "PRUNED" for t in tuner.study.trials)


def test_tester_evaluates_the_split_the_trainer_recorded(result_dir, tmp_path):
    geometry = ToyGeometry()
    context = PipelineContext(
        geometry=geometry, base_dir=str(tmp_path / "run"), result_dir_override=result_dir, seed=3
    )
    trainer = ModelTrainer(hyperparameters=TINY_MLP, test_frac=0.25, val_frac=0.25)

    context = trainer.run(context)

    split = pd.read_csv(context.split_csv_path)
    assert set(split["split"]) == {"train", "val", "test"}
    assert split["name"].is_unique
    test_names = set(split.loc[split["split"] == "test", "name"])
    assert context.test_df is not None
    assert test_names == set(context.test_df["name"])

    # A later pipeline: the trained model is available, but the split is not in memory
    later = PipelineContext(
        geometry=geometry, base_dir=str(tmp_path / "run"), result_dir_override=result_dir
    )
    later.trained_model = context.trained_model
    later.dataset = context.dataset
    tester = ModelTester()
    test_rows = tester._load_test_rows(later)
    assert test_rows is not None
    assert set(test_rows["name"]) == test_names

    later = tester.run(later)
    results = later.test_results
    assert results["n_samples"] == len(test_names)
    assert len(results["s_error_by_band"]) == 5  # the default band count
    assert list(results["worst_geometries"]) == sorted(
        results["worst_geometries"], key=results["worst_geometries"].get, reverse=True
    )
    assert results["s_error_percentiles"]["max"] >= results["s_error_percentiles"]["p50"]

    per_geometry = pd.read_csv(later.test_errors_csv_path)
    assert set(per_geometry["name"]) == test_names
    assert {"a", "b", "mean_abs_s_error", "max_abs_s_error"} <= set(per_geometry.columns)
    # k near zero would make a relative error meaningless, so it is reported as absolute
    assert "k abs error" in per_geometry.columns
    assert "k error %" not in per_geometry.columns

    profile = pd.read_csv(later.errors_vs_frequency_csv_path)
    assert {"|S|", "Lp", "k"} <= set(profile["parameter"])
    s_rows = profile[profile["parameter"] == "|S|"]
    assert len(s_rows) == len(FREQUENCIES)
    assert (s_rows["p5"] <= s_rows["median"]).all()
    assert (s_rows["median"] <= s_rows["p95"]).all()
    assert os.path.getsize(later.errors_vs_frequency_plot_path) > 0


def test_same_seed_gives_the_same_split_and_model(result_dir, tmp_path):
    losses = []
    for run in ("one", "two"):
        context = PipelineContext(
            geometry=ToyGeometry(),
            base_dir=str(tmp_path / run),
            result_dir_override=result_dir,
            seed=5,
        )
        context = ModelTrainer(hyperparameters=TINY_MLP).run(context)
        assert context.test_df is not None
        losses.append((sorted(context.test_df["name"]), context.final_val_loss))

    assert losses[0] == losses[1]


def test_batch_sizes_are_the_users_choice(result_dir):
    assert TrainingConfig.search_space()["batch_size"] == list(DEFAULT_BATCH_SIZES)

    tuner = HyperparameterTuner(OrcaMLP, _loaded(result_dir), batch_sizes=[7, 4096])

    assert tuner.search_space["batch_size"] == [7, 4096]


def test_epochs_are_not_searched():
    assert "epochs" not in TrainingConfig.search_space()


def test_tuner_reports_every_epoch_of_every_fold(result_dir):
    tuner = HyperparameterTuner(OrcaMLP, _loaded(result_dir), n_fold_cv=2, n_trials=1, max_epochs=3)

    tuner.tune()

    assert tuner.study is not None
    (trial,) = tuner.study.trials
    # Steps count epochs across folds: fold 1 is 0-2, fold 2 is 3-5
    assert sorted(trial.intermediate_values) == [0, 1, 2, 3, 4, 5]


def test_touchstone_files_are_parsed_once_per_run(result_dir, monkeypatch):
    parsed = []
    original = skrf.Network

    def counting_network(path, *args, **kwargs):
        parsed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr("orca.training.datasets.base_dataset.rf.Network", counting_network)
    prototype = _dataset(_iterator())
    table = _table(result_dir)

    prototype.new_split(result_dir, table, fit_normalizers=True)
    prototype.new_split(result_dir, table.head(4))
    assert len(parsed) == len(table)

    prototype.clear_cache()
    prototype.new_split(result_dir, table.head(4))
    assert len(parsed) == len(table) + 4


def test_tuning_timeout_stops_a_running_trial(result_dir, monkeypatch):
    tuner = HyperparameterTuner(OrcaMLP, _loaded(result_dir), n_fold_cv=2, n_trials=5, timeout=60)
    # The deadline is set at 0 s; every later reading is long past it
    clock = iter([0.0])
    monkeypatch.setattr("orca.training.tuner.time.monotonic", lambda: next(clock, 1e9))

    with pytest.raises(RuntimeError, match="still running at the tuning timeout"):
        tuner.tune()
    assert tuner.study is not None
    assert all(t.state.name == "PRUNED" for t in tuner.study.trials)
    # Each was pruned after its first epoch, before reporting anything
    assert not any(t.intermediate_values for t in tuner.study.trials)


def test_regularization_is_searched_only_when_asked(result_dir):
    dataset = _loaded(result_dir)

    plain = HyperparameterTuner(OrcaMLP, dataset).search_space
    regularized = HyperparameterTuner(OrcaMLP, dataset, regularization=True).search_space

    assert "weight_decay" not in plain
    assert "dropout" not in plain
    assert {"weight_decay", "dropout"} <= set(regularized)
    assert TrainingConfig.from_hyperparameters({"weight_decay": 0.5}).weight_decay == 0.5


def test_hyperparameters_are_saved_and_read_back(result_dir, tmp_path):
    first = PipelineContext(
        geometry=ToyGeometry(), base_dir=str(tmp_path / "one"), result_dir_override=result_dir, seed=5
    )
    first = ModelTrainer(hyperparameters=TINY_MLP).run(first)

    with open(first.hyperparameters_json_path) as f:
        assert json.load(f) == TINY_MLP

    second = PipelineContext(
        geometry=ToyGeometry(), base_dir=str(tmp_path / "two"), result_dir_override=result_dir, seed=5
    )
    second = ModelTrainer(hyperparameters=first.hyperparameters_json_path).run(second)

    assert second.hyperparameters == TINY_MLP
    assert second.final_val_loss == first.final_val_loss


def test_missing_hyperparameter_file_fails_before_loading(result_dir, tmp_path):
    context = PipelineContext(
        geometry=ToyGeometry(), base_dir=str(tmp_path), result_dir_override=result_dir
    )

    with pytest.raises(FileNotFoundError, match="neither a dict nor an existing JSON file"):
        ModelTrainer(hyperparameters=str(tmp_path / "missing.json")).run(context)


def _learning_rates(config: TrainingConfig, steps_per_epoch: int, n_steps: int) -> list[float]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=config.learning_rate)
    schedule = LearningRateSchedule(optimizer, config, steps_per_epoch)
    rates = []
    for _ in range(n_steps):
        schedule.before_step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


def test_learning_rate_warms_up_then_decays_along_a_cosine():
    config = TrainingConfig(epochs=4, learning_rate=1.0, warmup_epochs=1, device=torch.device("cpu"))

    rates = _learning_rates(config, steps_per_epoch=10, n_steps=40)

    assert rates[:10] == pytest.approx([(i + 1) / 10 for i in range(10)])
    assert rates[10] == pytest.approx(1.0)
    assert all(later < earlier for earlier, later in itertools.pairwise(rates[10:]))
    assert rates[-1] < 2 * config.min_lr_ratio


def test_plateau_schedule_starts_after_the_warmup():
    config = TrainingConfig(
        learning_rate=1.0,
        lr_schedule="plateau",
        warmup_epochs=1,
        scheduler_patience=0,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    schedule = LearningRateSchedule(optimizer, config, steps_per_epoch=4)

    for _ in range(2):  # Half the warmup: a worse validation loss changes nothing
        schedule.before_step()
    schedule.after_epoch(1.0)
    schedule.after_epoch(2.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    for _ in range(2):  # Warmup done: the plateau schedule takes over
        schedule.before_step()
    schedule.after_epoch(1.0)
    schedule.after_epoch(2.0)
    schedule.before_step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.scheduler_factor)


@pytest.mark.parametrize("max_norm", [0.5, None])
def test_gradients_are_clipped_to_the_configured_norm(result_dir, monkeypatch, max_norm):
    dataset = _loaded(result_dir)
    clipped_to = []
    clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, norm, *args, **kwargs):
        clipped_to.append(norm)
        return clip(parameters, norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    config = TrainingConfig.from_hyperparameters(
        {**TINY_MLP, "grad_clip_norm": max_norm}, device=dataset.device
    )
    Trainer(config, verbose=False).fit(
        OrcaMLP.from_spec(dataset.io_spec, TINY_MLP), dataset, dataset
    )

    assert set(clipped_to) == ({max_norm} if max_norm is not None else set())


def test_unknown_schedule_fails_when_the_stage_is_built():
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        ModelTrainer(lr_schedule="cosin")


def test_tuning_trials_use_the_stage_training_settings(result_dir):
    tuner = HyperparameterTuner(
        OrcaMLP,
        _loaded(result_dir),
        n_fold_cv=2,
        n_trials=1,
        training_defaults={"grad_clip_norm": -1.0},
    )

    with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
        tuner.tune()


def test_run_seed_seeds_every_global_generator(tmp_path):
    def draws() -> tuple[float, float, float]:
        return random.random(), float(np.random.rand()), float(torch.rand(1))  # noqa: S311, NPY002

    contexts = []
    runs = []
    for run in ("one", "two"):
        contexts.append(ORCA([]).run(geometry=ToyGeometry(), base_dir=str(tmp_path / run), seed=7))
        runs.append(draws())

    assert runs[0] == runs[1]
    assert all(context is not None and context.seed == 7 for context in contexts)
