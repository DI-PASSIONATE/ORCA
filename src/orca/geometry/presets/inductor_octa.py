from dataclasses import dataclass
from typing import Any
import numpy as np
import os
import time
import torch.nn as nn
import torchvision
import optuna
import skrf as rf
import onnxruntime

from orca import BaseGeometry
from orca.logger import logger
from orca.geometry.cells.inductor import symmetric_octa_IHP, get_min_outer_diameter
from orca.geometry.input_parameters import InputParameterIterator
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.normalize import (
    StandardNormalizer,
    OutputMinMaxNormalizer,
)
from orca.training.feature_transform import FeatureTransformPipeline
from orca.training.datasets.geo_to_s_param_single_f import (
    GeoToSParamDatasetSingleFrequency,
)
from orca.utils.postprocessing import s_param_dict_to_network

# Ground ring is drawn on Metal5 (matches "from_layername": "Metal5" in the
# simcfg). Feed is on TopMetal1, which requires N >= 2 turns (see
# symmetric_octa_IHP: N == 1 feeds on TopMetal2 instead).
GROUND_LAYER = 67       # Metal5
RING_SPACING = 20.0     # µm, gap between inductor outer edge and ground ring
RING_WIDTH = 10.0       # µm, ground-ring thickness


@dataclass
class InductorOcta(BaseGeometry):
    """
    Represents a symmetric octagonal spiral inductor geometry (IHP SG13G2).

    2-port (LA/LB) spiral inductor, ported from the gds2palace IHP example by
    Volker Muehlhaus. Requires N >= 2 turns, since the feedline sits on
    TopMetal1 (single-turn inductors feed on TopMetal2 instead).
    """

    name: str = "inductor_octa"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_nosub.xml")
    simconfig_filename: str = os.path.join(
        os.path.dirname(__file__), "inductor_octa.simcfg"
    )
    input_parameter_iterator: InputParameterIterator = InputParameterIterator(
        picking_strategy="random",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
        turns=[2, 3, 4, 5],
        width=[x / 100 for x in range(201, 1501, 1)],   # 2.01 .. 15.00 µm
        space=[x / 100 for x in range(201, 601, 1)],    # 2.01 ..  6.00 µm
        diameter=[float(x) for x in range(30, 301, 1)],  # 30 .. 300 µm
    )
    features = FeatureTransformPipeline(
        # RatioFeature(i=0, j=1),  # turns / width
        # ChebyshevFeature(i=4, degree=3),  # Chebyshev features of frequency
    )
    dataset: BaseDataset = GeoToSParamDatasetSingleFrequency(
        n_ports=2,
        features=features,
        input_normalizer=OutputMinMaxNormalizer(),
        output_normalizer=StandardNormalizer(),
    )

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "batch_size": [32, 64, 128, 256, 512],
            "epochs": optuna.distributions.IntDistribution(5, 50, step=5),
            "num_layers": optuna.distributions.IntDistribution(3, 9, step=1),
            "hidden_size": optuna.distributions.IntDistribution(128, 2048, step=128),
            "activation_function": ["GELU", "SiLU"],
        }

    def get_model(self, hyperparameters: dict[str, Any]) -> nn.Module:
        """
        Returns a new instance of the model with the specified hyperparameters.
        This allows for dynamic model creation during hyperparameter optimization.
        """
        print(f"Creating model with hyperparameters: {hyperparameters}")
        num_layers = hyperparameters["num_layers"]
        hidden_size = hyperparameters["hidden_size"]
        activation_function = hyperparameters["activation_function"]

        n_outputs = 2 * 2 * 2  # n_ports=2 -> 4 S-params x (Re + Im) = 8
        hidden_channels = [hidden_size] * num_layers + [n_outputs]
        return torchvision.ops.MLP(
            in_channels=4 + 1,  # turns, width, space, diameter + frequency
            hidden_channels=hidden_channels,
            activation_layer=getattr(nn, activation_function),
        )

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        N = int(round(params["turns"]))
        w = float(params["width"])
        s = float(params["space"])
        D = float(params["diameter"])

        # clamp the outer diameter to the minimum buildable (DRC-valid) value
        do_min = get_min_outer_diameter(N, w, s)
        if D < do_min:
            D = do_min

        symmetric_octa_IHP(
            N=N, D=D, w=w, s=s,
            includeCenterTap=False,
            LBE=False,
            forEM=True,
            ground_layer=GROUND_LAYER,
            ring_spacing=RING_SPACING,
            ring_width=RING_WIDTH,
            filename=output_path,
        )
        return output_path

    def inference_snp(
        self, onnx_session: onnxruntime.InferenceSession, input_params: np.ndarray
    ) -> rf.Network:
        """
        Runs inference on the model for the given geometry parameters and frequency points, and saves the predicted S-parameters to a Touchstone file.
        """
        t = time.time()
        frequency_points = np.arange(1e9, 201e9, 1e9)  # 1 GHz to 200 GHz

        # Create batched input by repeating the input parameters for each frequency point and adding the frequency as an additional feature
        batched_input = np.repeat(
            input_params[np.newaxis, :], len(frequency_points), axis=0
        )

        # Build feed_dict
        feed_dict: dict[str, Any] = {}

        # Process geometry parameters
        for i, param_name in enumerate(
            self.input_parameter_iterator.input_values.keys()
        ):
            feed_dict[param_name] = batched_input[:, i].reshape(-1, 1).astype(np.float32)

        # Process frequency
        feed_dict["frequency"] = frequency_points.reshape(-1, 1).astype(np.float32)

        # Run inference
        output_names = [node.name for node in onnx_session.get_outputs()]
        outputs = onnx_session.run(output_names, feed_dict)
        output_dict = dict(zip(output_names, outputs))

        t2 = time.time()
        logger.debug(
            f"Inference time in ms for {len(frequency_points)} frequency points: {(t2 - t) * 1000:.2f} ms"
        )

        _N, ntwk, _output_dict = s_param_dict_to_network(output_dict, frequency_points)

        return ntwk
