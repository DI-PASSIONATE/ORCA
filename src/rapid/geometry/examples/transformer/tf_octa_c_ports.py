from rapid.geometry.base_geometry import BaseGeometry

class Transformer(BaseGeometry):
    def __init__(self, name, stackup_xml, simconfig_filename):
        super().__init__(name, stackup_xml, simconfig_filename)
        self.n_inputs = 0
        self.n_outputs = 4

    def create_gds_file(self, input_parameters):
        super().create_gds_file(input_parameters)
        # Implement geometry creation logic here

        # TODO: Replace with actual geometry creation code
        return "/home/david/Documents/git/RAPID/src/rapid/geometry/examples/transformer/REFERENCE.gds"
    
    def get_next_input_parameters(self):
        return None