from dataclasses import dataclass
from typing import Any
import numpy as np
import os
import torch.nn as nn
import torchvision

from orca import BaseGeometry
from orca.geometry.cells.transformer import tf_octa_c
from orca.geometry.input_parameters import InputParameterIterator
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.normalize import MinMaxNormalizer, StandardNormalizer, OutputMinMaxNormalizer
from orca.utils.postprocessing import *

from orca.training.models.mlp import OrcaMLP
from orca.training.normalize import MinMaxNormalizer
from orca.training.feature_transform import FeatureTransformPipeline, RatioFeature, ChebyshevFeature
from orca.training.datasets.geo_to_s_param_single_f import GeoToSParamDatasetSingleFrequency

@dataclass
class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape.
    """
    name: str = "tf_octa_c_ports"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_nosub.xml")
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
    params = None
    input_parameter_iterator: InputParameterIterator = InputParameterIterator(
        picking_strategy="random",
        frequency=[1e8, 200e8],  # 1 GHz to 200 GHz
        input_winding_diameter = [x/10 for x in range(200, 1001, 1)], # 20.0 to 100.0 in 0.1 steps
        output_winding_diameter = [x/10 for x in range(200, 1001, 1)], # 20.0 to 100.0 in 0.1 steps
        center_displacement = [x/10 for x in range(0, 151, 1)], # 0.0 to 15.0 in 0.1 steps
        bottom_linewidth = [x/10 for x in range(20, 81, 1)], # 2.0 to 10.0 in 0.1 steps
        upper_linewidth = [x/10 for x in range(20, 81, 1)], # 2.0 to 10.0 in 0.1 steps
    )
    features = FeatureTransformPipeline(
        RatioFeature(i=0, j=1),  # input_winding_diameter / output_winding_diameter
        RatioFeature(i=3, j=4),  # bottom_linewidth / upper_linewidth
        RatioFeature(i=5, j=0),  # frequency / input_winding_diameter
        ChebyshevFeature(i=5, degree=3),  # Chebyshev features of frequency
    )
    dataset: BaseDataset = GeoToSParamDatasetSingleFrequency(
        n_ports=6,
        features=features,
        input_normalizer=OutputMinMaxNormalizer(),
        output_normalizer=StandardNormalizer(),
    )
    model: nn.Module = torchvision.ops.MLP(
        in_channels=5+1+len(features),  # 5 original params + 1 frequency + features
        hidden_channels=[128, 256, 256, 128, 72],  # 6-port S-parameters (Re/Im) -> 6*6*2=72
        activation_layer=nn.SiLU
    )

    @staticmethod
    def create_gds_file(name:str, output_path:str, params: dict[str, Any]) -> str:
        c = tf_octa_c(
            name=name,
            input_winding_diameter=params["input_winding_diameter"],
            output_winding_diameter=params["output_winding_diameter"],
            center_displacement=params["center_displacement"],
            bottom_linewidth=params["bottom_linewidth"],
            upper_linewidth=params["upper_linewidth"],
            bottom_center_tap_width=0,
            upper_center_tap_width=0,
            lower_feed_type=1,
            upper_feed_type=1,
            feedline_spacing=max(params["bottom_linewidth"], params["upper_linewidth"]) + 0.1,
            gnd_upper_spacing=20,
            gnd_lower_spacing=20,
            gnd_side_spacing=20,
            gnd_ring_width=10,
        )
        #c.show()
        c.write_gds(output_path, with_metadata=False)
        return output_path

    def postprocess_outputs(self, output, frequency_points=None):
        """
        Converts model outputs (Re/Im) into a .sNp Touchstone file format.
        Plots the S-parameters for visualization.

        Parameters
        ----------
        output : dict
            Dictionary containing S-parameters split into real and imaginary parts.
            Example keys: 'S11_real', 'S11_imag', ..., 'SNN_real', 'SNN_imag'.
            Each value is a 1D array of length equal to len(f).
        f : array-like
            1D array of frequencies corresponding to the S-parameters.
        filename : str, optional
            Name of the Touchstone file to save, default "output.sNp".
        """
        # Frequency points are just from 1 to 200 in 1 GHz steps
        if frequency_points is None:
            frequency_points = np.arange(1, 201)  # 1 GHz to 200 GHz
        N, ntwk, output_dict = s_param_dict_to_network(output, frequency_points)
        filename = f"{self.name}.s{N}p"
        ntwk.write_touchstone(filename)

        #N, ntwk = single_ended_to_mixed_mode(ntwk)
        plot_rfic_transformer_metrics(ntwk)
        # plot_diff_s_params_and_k(ntwk)

        # Write Touchstone
        print(f"Touchstone file saved as {filename}")
        
        return output_dict