from orca import BaseGeometry
import os

class Transformer(BaseGeometry):
    def __init__(self, name, stackup_xml, simconfig_filename):
        super().__init__(name, stackup_xml, simconfig_filename)
        self.n_inputs = 0
        self.n_outputs = 4

    def create_gds_file(self):
        super().create_gds_file()
        # Implement geometry creation logic here

        # TODO: Replace with actual geometry creation code
        return f"{os.path.dirname(__file__)}/REFERENCE.gds"
    
    def get_next_input_parameters(self):
        return None