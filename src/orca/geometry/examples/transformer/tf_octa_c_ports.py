from orca import BaseGeometry
import os
from .transformer_factory import tf_octa_c

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

        self.n_inputs = 0
        self.n_outputs = 4
    
    def get_next_input_parameters(self):
        return None

    def create_gds_file(self):
        super().create_gds_file()

        output_path = os.path.join(
            os.path.dirname(__file__),
            self.name + ".gds"
        )

        # TODO: Replace with actual geometry creation code
        # return f"{os.path.dirname(__file__)}/tf_octa_c.gds"
        c = tf_octa_c(
            input_winding_diameter=65.0,
            output_winding_diameter=65.0,
            center_displacement=10.0,
            bottom_linewidth=7.0,
            upper_linewidth=7.0,
            bottom_center_tap_width=0.0,
            upper_center_tap_width=0.0,
            lower_feed_type=1,
            upper_feed_type=1,
            feedline_spacing=7.0,
            gnd_upper_spacing=20.0,
            gnd_lower_spacing=20.0,
            gnd_side_spacing=20.0,
            gnd_ring_width=10.0,
        )
        c.show()
        c.write_gds(output_path)
        return output_path
    