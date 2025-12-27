import gdsfactory as gf

class BaseGeometry:
    def __init__(self, stackup_xml: str, n_inputs: int, n_outputs: int):
        self.stackup_xml = stackup_xml
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def create_geometry(self, input_parameters: list) -> gf.Component:
        assert len(input_parameters) == self.n_inputs, f"Expected {self.n_inputs} input parameters, got {len(input_parameters)}"
    
    def simulation_output_to_scalar(self, simulation_output: list) -> int:
        assert len(simulation_output) == self.n_outputs, f"Expected {self.n_outputs} output values, got {len(simulation_output)}"