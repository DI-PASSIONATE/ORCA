"""Loss terms, self-resonance detection and sample weighting (no Palace needed)."""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
skrf = pytest.importorskip("skrf")

from orca.geometry.base_geometry import BaseGeometry
from orca.geometry.input_parameters import InputParameterIterator, RangeParameter
from orca.pipeline.context import PipelineContext
from orca.pipeline.training_stage import ModelTrainer
from orca.training.codecs import FlatReImCodec, UpperTriangleReImCodec
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.datasets.geo_to_s_param_single_f import (
    GeoToSParamDatasetSingleFrequency,
)
from orca.training.losses import (
    SParameterLoss,
    admittance_error,
    first_self_resonance,
    passivity_violation,
    s_to_y,
    srf_sample_weights,
)
from orca.training.normalize import MinMaxNormalizer, StandardNormalizer
from orca.training.trainer import make_batches

# From DC, so the DC point that dc_deembedded results start with is covered too
FREQUENCIES = np.linspace(0.0, 10e9, 41)
L_NH = [1.0, 2.0, 3.0, 4.0]
C_PF = [0.5, 1.0, 1.5]
R_SERIES = 1.0
Z0 = 50.0


def _pi_network(inductance: float, capacitance: float, frequencies=FREQUENCIES) -> np.ndarray:
    """S-matrices of a series R-L between two ports, with a shunt C at each port."""
    omega = 2 * np.pi * frequencies
    y_series = 1 / (R_SERIES + 1j * omega * inductance)
    y_shunt = 1j * omega * capacitance
    y = np.empty((len(frequencies), 2, 2), dtype=complex)
    y[:, 0, 0] = y[:, 1, 1] = y_series + y_shunt
    y[:, 0, 1] = y[:, 1, 0] = -y_series
    eye = np.eye(2)
    return (eye - Z0 * y) @ np.linalg.inv(eye + Z0 * y)


def _resonance(inductance: float, capacitance: float) -> float:
    """Where the differential susceptance Im(2 Y_series + Y_shunt) crosses zero."""
    return math.sqrt(2 / (inductance * capacitance) - (R_SERIES / inductance) ** 2) / (2 * math.pi)


def _iterator() -> InputParameterIterator:
    return InputParameterIterator(
        RangeParameter("a", L_NH[0], L_NH[-1], step=1.0),
        RangeParameter("b", C_PF[0], C_PF[-1], step=0.5),
        frequency=[FREQUENCIES[0], FREQUENCIES[-1]],
    )


def _iterator_normalizer() -> MinMaxNormalizer:
    return MinMaxNormalizer(_iterator())


def _dataset() -> GeoToSParamDatasetSingleFrequency:
    return GeoToSParamDatasetSingleFrequency(
        codec=FlatReImCodec(n_ports=2),
        input_normalizer=_iterator_normalizer(),
        output_normalizer=StandardNormalizer(),
    )


@dataclass
class ResonatorGeometry(BaseGeometry):
    name: str = "resonator"
    stackup_xml: str = ""
    simconfig_filename: str = ""
    input_parameter_iterator: InputParameterIterator = field(default_factory=_iterator)

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict) -> str:
        raise NotImplementedError

    def create_dataset(self) -> GeoToSParamDatasetSingleFrequency:
        return _dataset()


@pytest.fixture
def result_dir(tmp_path) -> str:
    """One pi-network per (L, C) pair; the smallest L*C resonates just above the band."""
    rows = []
    for i, (a, b) in enumerate((a, b) for a in L_NH for b in C_PF):
        s = _pi_network(a * 1e-9, b * 1e-12)
        ntwk = skrf.Network(frequency=skrf.Frequency.from_f(FREQUENCIES, unit="hz"), s=s)
        ntwk.write_touchstone(filename=f"res_{i}", dir=str(tmp_path))
        rows.append({"name": f"res_{i}.s2p", "a": a, "b": b})
    pd.DataFrame(rows).to_csv(tmp_path / "resonator.csv", index=False)
    return str(tmp_path)


def _loaded(result_dir: str) -> BaseDataset:
    table = pd.read_csv(os.path.join(result_dir, "resonator.csv"))
    return _dataset().new_split(result_dir, table, fit_normalizers=True)


@pytest.mark.parametrize("codec", [FlatReImCodec(n_ports=3), UpperTriangleReImCodec(n_ports=3)])
def test_tensor_decode_matches_numpy_decode(codec):
    raw = np.random.default_rng(0).normal(size=(5, codec.output_dim)).astype(np.float32)

    decoded = codec.decode_tensor(torch.from_numpy(raw)).numpy()

    np.testing.assert_array_equal(decoded, codec.decode(raw))


def test_admittance_matches_scikit_rf():
    s = _pi_network(2e-9, 1e-12)
    ntwk = skrf.Network(frequency=skrf.Frequency.from_f(FREQUENCIES, unit="hz"), s=s, z0=Z0)

    y = s_to_y(torch.from_numpy(s)).numpy()

    np.testing.assert_allclose(y, ntwk.y * Z0, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("inductance", "capacitance"), [(1e-9, 1e-12), (4e-9, 1.5e-12)])
def test_self_resonance_of_a_pi_network(inductance, capacitance):
    s = _pi_network(inductance, capacitance)
    shuffled = np.random.default_rng(1).permutation(len(FREQUENCIES))

    found = first_self_resonance(FREQUENCIES[shuffled], s[shuffled])

    assert found == pytest.approx(_resonance(inductance, capacitance), rel=1e-2)


def test_no_self_resonance_in_the_band_is_infinite():
    # 1 nH and 0.5 pF resonate at about 10.07 GHz, just above the band
    assert first_self_resonance(FREQUENCIES, _pi_network(1e-9, 0.5e-12)) == math.inf


def test_points_above_the_resonance_are_weighted_apart(result_dir):
    dataset = _loaded(result_dir)

    weights = srf_sample_weights(dataset, above_srf_weight=0.1).cpu().numpy()

    assert weights.mean() == pytest.approx(1.0, rel=1e-6)
    frequencies = _iterator_normalizer().denormalize(dataset.inputs.cpu())[:, -1].numpy()
    table = pd.read_csv(os.path.join(result_dir, "resonator.csv")).set_index("name")
    groups = np.asarray(dataset.sample_groups)
    below_weight = weights[frequencies <= 1e9].max()
    for name, (a, b) in table[["a", "b"]].iterrows():
        mine = groups == name
        above = frequencies[mine] > _resonance(a * 1e-9, b * 1e-12)
        expected = np.where(above, 0.1 * below_weight, below_weight)
        np.testing.assert_allclose(weights[mine], expected, rtol=1e-5)
    # 1 nH, 0.5 pF has no resonance in the band and keeps full weight everywhere
    assert np.all(weights[groups == "res_0.s2p"] == below_weight)


def test_without_terms_the_loss_is_the_data_loss():
    pred, target = torch.randn(6, 8), torch.randn(6, 8)
    loss = SParameterLoss(torch.nn.L1Loss(), FlatReImCodec(n_ports=2))

    assert loss(pred, target) == pytest.approx(torch.nn.functional.l1_loss(pred, target).item())

    weights = torch.tensor([3.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    per_sample = (pred - target).abs().mean(dim=1)
    assert loss(pred, target, weights) == pytest.approx((per_sample * weights).mean().item())


def test_loss_needs_an_elementwise_data_loss():
    with pytest.raises(TypeError, match="reduction"):
        SParameterLoss(lambda pred, target: (pred - target).abs().mean(), FlatReImCodec(2))


def test_admittance_error_sees_a_loss_error_that_s_hides():
    s_true = torch.from_numpy(_pi_network(2e-9, 1e-12)[1:5])  # 0.25 to 1 GHz
    # Twice the series resistance: Q halves, but S barely moves
    eye = np.eye(2)
    omega = 2 * np.pi * FREQUENCIES[1:5]
    y_series = 1 / (2 * R_SERIES + 1j * omega * 2e-9)
    y = np.empty((4, 2, 2), dtype=complex)
    y[:, 0, 0] = y[:, 1, 1] = y_series + 1j * omega * 1e-12
    y[:, 0, 1] = y[:, 1, 0] = -y_series
    s_lossier = torch.from_numpy((eye - Z0 * y) @ np.linalg.inv(eye + Z0 * y))

    assert torch.all(admittance_error(s_true, s_true) == 0)
    assert torch.all((s_lossier - s_true).abs().amax(dim=(1, 2)) < 0.05)
    assert torch.all(admittance_error(s_lossier, s_true) > 0.5)


def test_passivity_is_only_penalised_beyond_the_reference():
    eye = torch.eye(2, dtype=torch.complex64).expand(3, 2, 2)
    s_true = eye * torch.tensor([0.5, 1.0, 1.1]).view(3, 1, 1)
    s_pred = eye * torch.tensor([0.9, 1.2, 1.1]).view(3, 1, 1)

    violation = passivity_violation(s_pred, s_true)

    np.testing.assert_allclose(violation.numpy(), [0.0, 0.2, 0.0], atol=1e-6)


def test_sample_weights_are_batched_with_their_samples(result_dir):
    dataset = _loaded(result_dir)
    dataset.set_sample_weights(torch.arange(len(dataset), dtype=torch.float32, device=dataset.device))
    subset = torch.utils.data.Subset(dataset, [5, 2, 7])

    (x, _, w), *_ = list(make_batches(subset, batch_size=10, shuffle=False))

    assert torch.equal(x, dataset.inputs[[5, 2, 7]])
    assert w.tolist() == [5.0, 2.0, 7.0]
    with pytest.raises(ValueError, match="one weight per sample"):
        dataset.set_sample_weights(torch.ones(3))


def test_stage_trains_with_every_loss_option(result_dir, tmp_path):
    context = PipelineContext(
        geometry=ResonatorGeometry(),
        base_dir=str(tmp_path / "run"),
        result_dir_override=result_dir,
        seed=3,
    )
    trainer = ModelTrainer(
        n_fold_cv=1,
        n_trials=1,
        tuning_max_epochs=2,
        max_epochs=2,
        batch_sizes=[16],
        admittance_weight=0.5,
        passivity_weight=1.0,
        above_srf_weight=0.1,
    )

    context = trainer.run(context)

    assert context.final_val_loss is not None
    assert math.isfinite(context.final_val_loss)


@pytest.mark.parametrize(
    "options",
    [{"admittance_weight": -1.0}, {"passivity_weight": -0.1}, {"above_srf_weight": 0.0}],
)
def test_invalid_loss_options_fail_when_the_stage_is_built(options):
    with pytest.raises(ValueError, match="must"):
        ModelTrainer(**options)
