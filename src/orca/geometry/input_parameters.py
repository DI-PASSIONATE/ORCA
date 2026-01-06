from dataclasses import dataclass, field

@dataclass
class InputParameters:
    """Class for keeping track of geometry input parameters."""

    n_inputs: int
    input_values: dict[str, float|int] = field(default_factory=dict)

    def __post_init__(self):
        assert self.n_inputs == len(self.input_values), f"n_inputs ({self.n_inputs}) must match length of input_values ({len(self.input_values)})"