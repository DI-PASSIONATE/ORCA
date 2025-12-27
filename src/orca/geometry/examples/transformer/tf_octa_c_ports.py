from orca import BaseGeometry
import os

class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape and C-ports.
    """

    def __init__(self):
        # These files are already included in the repository
        NAME = "tf_octa_c_ports"
        STACKUP_XML = os.path.join(os.path.dirname(__file__), "..", "SG13G2_nosub.xml")
        SIMCONFIG_FILENAME = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")

        # Call the base class constructor with the parameters
        super().__init__(NAME, STACKUP_XML, SIMCONFIG_FILENAME)

        self.n_inputs = 0
        self.n_outputs = 4

    def create_gds_file(self):
        super().create_gds_file()
        # Implement geometry creation logic here

        # TODO: Replace with actual geometry creation code
        return f"{os.path.dirname(__file__)}/REFERENCE.gds"
    
    def get_next_input_parameters(self):
        return None