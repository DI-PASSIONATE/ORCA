from dataclasses import dataclass
from typing import Any
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import optuna
import skrf as rf
import onnxruntime
import onnx

from orca import BaseGeometry
from orca.logger import logger
from orca.geometry.cells.transformer import tf_octa_c
from orca.geometry.input_parameters import InputParameterIterator
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.normalize import (
    StandardNormalizer,
    OutputMinMaxNormalizer,
)
from orca.utils.postprocessing import *

from orca.training.feature_transform import (
    FeatureTransformPipeline,
    RatioFeature,
    ChebyshevFeature,
)
from orca.training.datasets.geo_to_s_param_single_f import (
    GeoToSParamDatasetSingleFrequency,
)

# import gdstk
# def _snap_gds_inplace(path: str, grid_nm: int = 10) -> None:
#     """Snap all polygon vertices in a GDS file to the nearest grid_nm grid.

#     gf.Path.extrude() produces off-grid vertices for angled octagon segments
#     (e.g. width/2 * sin(22.5°) = 0.957 µm is not on the 10 nm grid).
#     gdstk is always available as a gdsfactory dependency.
#     """
#     grid_um = grid_nm / 1000.0
#     lib = gdstk.read_gds(path)
#     for cell in lib.cells:
#         for poly in cell.polygons:
#             poly.points = np.round(poly.points / grid_um) * grid_um
#     lib.write_gds(path)

@dataclass
class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape.
    """

    name: str = "tf_octa_c_ports"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_nosub.xml")
    simconfig_filename: str = os.path.join(
        os.path.dirname(__file__), "tf_octa_c_ports.simcfg"
    )
    input_parameter_iterator: InputParameterIterator = InputParameterIterator(
        picking_strategy="random",
        frequency=[1e8, 500e8],  # 1 GHz to 500 GHz
        bottom_winding_diameter=[
            x / 10 for x in range(200, 1201, 1)
        ],  # 20.0 to 120.0 in 0.1 steps
        top_winding_diameter=[
            x / 10 for x in range(200, 1201, 1)
        ],  # 20.0 to 120.0 in 0.1 steps
        center_displacement=[
            x / 10 for x in range(0, 151, 1)
        ],  # 0.0 to 15.0 in 0.1 steps
        bottom_linewidth=[x / 10 for x in range(20, 121, 1)],  # 2.0 to 12.0 in 0.1 steps
        top_linewidth=[x / 10 for x in range(20, 121, 1)],  # 2.0 to 12.0 in 0.1 steps
    )
    features = FeatureTransformPipeline(
        # RatioFeature(i=0, j=1),  # input_winding_diameter / output_winding_diameter
        # RatioFeature(i=3, j=4),  # bottom_linewidth / upper_linewidth
        # RatioFeature(i=5, j=0),  # frequency / input_winding_diameter
        # ChebyshevFeature(i=5, degree=3),  # Chebyshev features of frequency
    )
    dataset: BaseDataset = GeoToSParamDatasetSingleFrequency(
        n_ports=6,
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
            #"dropout": optuna.distributions.FloatDistribution(0.0, 0.4, step=0.1),
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
        #dropout = hyperparameters["dropout"]
        activation_function = hyperparameters["activation_function"]

        hidden_channels = [hidden_size] * num_layers + [72]  # 72 output channels for 6 ports
        return torchvision.ops.MLP(
            in_channels=5 + 1,# + len(self.features),
            hidden_channels=hidden_channels,
            activation_layer=getattr(nn, activation_function),
            #dropout=dropout,
        )

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        c = tf_octa_c(
            name=name,
            bottom_winding_diameter=params["bottom_winding_diameter"],
            top_winding_diameter=params["top_winding_diameter"],
            center_displacement=params["center_displacement"],
            bottom_linewidth=params["bottom_linewidth"],
            top_linewidth=params["top_linewidth"],
            bottom_center_tap_width=0,
            upper_center_tap_width=0,
            lower_feed_type=1,
            upper_feed_type=1,
            feedline_spacing=max(params["bottom_linewidth"], params["top_linewidth"])
            + 0.1,
            gnd_upper_spacing=20,
            gnd_lower_spacing=20,
            gnd_side_spacing=20,
            gnd_ring_width=10,
        )
        # c.show()
        c.write_gds(output_path, with_metadata=False)
        return output_path
    
    def inference_snp(self, onnx_session: onnxruntime.InferenceSession, input_params: np.ndarray) -> rf.Network:
        """
        Runs inference on the model for the given geometry parameters and frequency points, and saves the predicted S-parameters to a Touchstone file.
        """
        # Create frequency points from 1 GHz to 200 GHz in 1 GHz steps
        import time
        t = time.time()
        frequency_points = np.arange(0, 201e9, 1e9)

        # Create batched input by repeating the input parameters for each frequency point and adding the frequency as an additional feature
        batched_input = np.repeat(input_params[np.newaxis, :], len(frequency_points), axis=0)
        
        # Build feed_dict
        feed_dict = {}
        
        # Process geometry parameters
        for i, param_name in enumerate(self.input_parameter_iterator.input_values.keys()):
            feed_dict[param_name] = batched_input[:, i].reshape(-1, 1).astype(np.float32)
        
        # Process frequency
        feed_dict["frequency"] = (frequency_points).reshape(-1, 1).astype(np.float32)
        
        # Run inference
        output_names = [node.name for node in onnx_session.get_outputs()]

        # Actual inference
        outputs = onnx_session.run(output_names, feed_dict)
        output_dict = dict(zip(output_names, outputs))

        t2 = time.time()
        logger.debug(f"Inference time in ms for {len(frequency_points)} frequency points: {(t2 - t) * 1000:.2f} ms")

        N, ntwk, output_dict = s_param_dict_to_network(output_dict, frequency_points)

        return ntwk

    # def postprocess_outputs(self, output, frequency_points=None):
    #     """
    #     Converts model outputs (Re/Im) into a .sNp Touchstone file format.
    #     Plots the S-parameters for visualization.

    #     Parameters
    #     ----------
    #     output : dict
    #         Dictionary containing S-parameters split into real and imaginary parts.
    #         Example keys: 'S11_real', 'S11_imag', ..., 'SNN_real', 'SNN_imag'.
    #         Each value is a 1D array of length equal to len(f).
    #     f : array-like
    #         1D array of frequencies corresponding to the S-parameters.
    #     filename : str, optional
    #         Name of the Touchstone file to save, default "output.sNp".
    #     """
    #     # Frequency points are just from 1 to 200 in 1 GHz steps
    #     if frequency_points is None:
    #         frequency_points = np.arange(1, 201)  # 1 GHz to 200 GHz
    #     N, ntwk, output_dict = s_param_dict_to_network(output, frequency_points)
    #     filename = f"{self.name}.s{N}p"
    #     ntwk.write_touchstone(filename)

    #     # N, ntwk = single_ended_to_mixed_mode(ntwk)
    #     plot_rfic_transformer_metrics(ntwk)
    #     # plot_diff_s_params_and_k(ntwk)

    #     # Write Touchstone
    #     print(f"Touchstone file saved as {filename}")

    #     return output_dict

# if __name__ == "__main__":
#     geometry = TransformerOcta()
#     input_params = np.array([70.6, 74.6, 13.2, 6.4, 5.4])  # Example input parameters
#     onnx_session = onnxruntime.InferenceSession("/home/david/Documents/git/ORCA/output/tf_octa_c_ports/models/tf_octa_c_ports.onnx")
#     ntwk = geometry.inference_snp(onnx_session, input_params)
#     plot_rfic_transformer_metrics(ntwk)