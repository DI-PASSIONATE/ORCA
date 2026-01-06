from orca import BaseGeometry
from orca.geometry.input_parameters import InputParameters
import os
from .transformer_factory import tf_octa_c

LAYER_TOP = (134, 0)   # TM2: Top Winding
LAYER_BOT = (126, 0)    # TM1: Bottom Winding
LAYER_RING = (67, 0)  # Ground Ring

class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape and C-ports.
    No input parameters are required for this geometry.
    """

    def __init__(self,
                 name = "tf_octa_c_ports",
                 stackup_xml: str = os.path.join(os.path.dirname(__file__), "..", "SG13G2_nosub.xml"),
                 simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
                ):

        # Call the base class constructor with the parameters
        super().__init__(name, stackup_xml, simconfig_filename)

        self.n_outputs = 4
        self.n_inputs = 14
        self.input_names = [
            "input_winding_diameter",
            "output_winding_diameter",
            "center_displacement",
            "bottom_linewidth",
            "upper_linewidth",
            "bottom_center_tap_width",
            "upper_center_tap_width",
            "lower_feed_type",
            "upper_feed_type",
            "feedline_spacing",
            "gnd_upper_spacing",
            "gnd_lower_spacing",
            "gnd_side_spacing",
            "gnd_ring_width"
        ]
    
    def get_next_input_parameters(self) -> InputParameters:
        # For testing purposes, return fixed values
        return InputParameters(
            n_inputs=self.n_inputs,
            input_names=self.input_names,
            input_values={
                "input_winding_diameter": 65.0,
                "output_winding_diameter": 65.0,
                "center_displacement": 10.0,
                "bottom_linewidth": 7.0,
                "upper_linewidth": 7.0,
                "bottom_center_tap_width": 0.0,
                "upper_center_tap_width": 0.0,
                "lower_feed_type": 1,
                "upper_feed_type": 1,
                "feedline_spacing": 7.0,
                "gnd_upper_spacing": 20.0,
                "gnd_lower_spacing": 20.0,
                "gnd_side_spacing": 20.0,
                "gnd_ring_width": 10.0
            }
        )

    def create_gds_file(self) -> str:
        output_path = os.path.join(
            os.path.dirname(__file__),
            self.name + ".gds"
        )

        print(f"Creating GDS file at: {output_path}")

        c = tf_octa_c(
            input_winding_diameter=self.input_parameters.input_values["input_winding_diameter"],
            output_winding_diameter=self.input_parameters.input_values["output_winding_diameter"],
            center_displacement=self.input_parameters.input_values["center_displacement"],
            bottom_linewidth=self.input_parameters.input_values["bottom_linewidth"],
            upper_linewidth=self.input_parameters.input_values["upper_linewidth"],
            bottom_center_tap_width=self.input_parameters.input_values["bottom_center_tap_width"],
            upper_center_tap_width=self.input_parameters.input_values["upper_center_tap_width"],
            lower_feed_type=self.input_parameters.input_values["lower_feed_type"],
            upper_feed_type=self.input_parameters.input_values["upper_feed_type"],
            feedline_spacing=self.input_parameters.input_values["feedline_spacing"],
            gnd_upper_spacing=self.input_parameters.input_values["gnd_upper_spacing"],
            gnd_lower_spacing=self.input_parameters.input_values["gnd_lower_spacing"],
            gnd_side_spacing=self.input_parameters.input_values["gnd_side_spacing"],
            gnd_ring_width=self.input_parameters.input_values["gnd_ring_width"],
        )
        c.show()
        c.write_gds(output_path)
        return output_path