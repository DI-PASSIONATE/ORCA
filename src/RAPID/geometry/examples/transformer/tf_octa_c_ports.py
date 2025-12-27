from RAPID.geometry.base_geometry import BaseGeometry

class Transformer(BaseGeometry):
    def create_gds_file(self, input_parameters):
        super().create_gds_file(input_parameters)
        # Implement geometry creation logic here

        # TODO: Replace with actual geometry creation code
        return "/home/david/Documents/git/RAPID/src/RAPID/geometry/examples/transformer/REFERENCE.gds"
    
    def get_next_input_parameters(self):
        return None