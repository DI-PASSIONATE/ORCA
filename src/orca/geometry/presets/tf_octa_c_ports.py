import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orca import BaseGeometry
from orca.geometry.cells.transformer import tf_octa_c
from orca.geometry.input_parameters import InputParameterIterator

if TYPE_CHECKING:
    from orca.training.datasets.base_dataset import BaseDataset

# Built per instance rather than shared as a class attribute: a dataclass default
# holds one object for every instance of the class, so two geometries would share
# one iterator. The dataset is built per instance too, by create_dataset().


def _input_parameters() -> InputParameterIterator:
    return InputParameterIterator(
        picking_strategy="random",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
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


@dataclass
class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape.
    """

    name: str = "tf_octa_c_ports"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_200um.xml")
    simconfig_filename: str = os.path.join(
        os.path.dirname(__file__), "tf_octa_c_ports.simcfg"
    )
    input_parameter_iterator: InputParameterIterator = field(
        default_factory=_input_parameters
    )

    def create_dataset(self) -> "BaseDataset":
        # Imported here so the geometry can be drawn and simulated without the
        # "train" extra (PyTorch) installed.
        from orca.training.codecs import FlatReImCodec
        from orca.training.datasets.geo_to_s_param_single_f import (
            GeoToSParamDatasetSingleFrequency,
        )
        from orca.training.normalize import MinMaxNormalizer, StandardNormalizer

        return GeoToSParamDatasetSingleFrequency(
            codec=FlatReImCodec(n_ports=6),
            # Scale inputs with the declared parameter ranges rather than the
            # min/max of whatever subset happens to be trained on.
            input_normalizer=MinMaxNormalizer(self.input_parameter_iterator),
            output_normalizer=StandardNormalizer(),
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
            + 5,
            gnd_upper_spacing=40,
            gnd_lower_spacing=40,
            gnd_side_spacing=40,
            gnd_ring_width=20,
        )
        c.write_gds(output_path, with_metadata=False)
        return output_path
