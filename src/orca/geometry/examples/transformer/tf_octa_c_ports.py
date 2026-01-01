from orca import BaseGeometry
import os

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
        # Implement geometry creation logic here

        # TODO: Replace with actual geometry creation code
        return f"{os.path.dirname(__file__)}/tf_octa_c.gds"
    