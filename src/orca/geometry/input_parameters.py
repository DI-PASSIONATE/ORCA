import threading
from itertools import product
import numpy as np

from orca import MIN_FREQUENCY, MAX_FREQUENCY

class InputParameterIterator:
    """
    Class for keeping track of geometry input parameters.
    
    Attributes:
        input_values (dict): Dictionary mapping parameter names to their values or ranges.
    """

    def __init__(self, n_samples: int = 1, picking_strategy: str = "grid", **input_values):
        """
        Initializes InputParameters with given input values and picking strategy.   

        Args:
            picking_strategy (str): Strategy for picking parameters ('grid', 'random', etc.).
            **input_values: Keyword arguments representing parameter names and their possible ranges, e.g. radius_bottom=range(20, 100). Possible values can be lists, ranges, or numpy arrays.
        """
        # Check if all input_values are lists or ranges
        for name, values in input_values.items():
            if not isinstance(values, (list, range, np.ndarray)):
                raise ValueError(f"Input kwarg '{name}' must be a list, range, or numpy array of possible values. Found type: {type(values)} (value: {values})")

        self.n_samples = n_samples
        self.picking_strategy = picking_strategy
        self.n_inputs = len(input_values)
        self.input_values = input_values
        self.input_names = list(input_values.keys())
        
        self._lock = threading.Lock() # For thread-safe iteration

        if self.picking_strategy == "step_grid":
            self._iterator = self.step_grid()
        elif self.picking_strategy == "uniform_grid" or self.picking_strategy == "grid":
            self._iterator = self.uniform_grid()
        else:
            raise ValueError(f"Unsupported picking strategy: {self.picking_strategy}. Supported strategies are 'grid'.")

    def __len__(self):
        """
        THIS RETURNS THE AMOUNT OF INPUT PARAMETERS, NOT THE AMOUNT OF GEOMETRIES THAT CAN BE GENERATED!
        """
        return len(self.input_values)        

    def __iter__(self):
        self.n_geometries_created = 0
        return self
    
    def __next__(self) -> dict[str, any]:
        with self._lock: # Ensure thread-safe access
            print("Getting next set of input parameters...")
            self.n_geometries_created += 1 # May be used for logging or tracking
            try:
                params = next(self._iterator) # Raises StopIteration when exhausted, which is propagated to this iterator
                # Convert numpy types to Python native types to avoid type issues with downstream libraries
                params = [param.item() if isinstance(param, np.generic) else param for param in params]
                return dict(zip(self.input_names, params))
            except StopIteration:
                raise

    def get_min_max_values(self, add_frequency_dim=False) -> tuple[list[float], list[float]]:
        """
        Returns the minimum and maximum values for each input parameter.
        Useful for normalization purposes.
        Returns:
            tuple: A tuple containing two lists - (min_values, max_values).
        """
        return (
            [min(param_list ) for param_list in self.input_values.values()] + ([MIN_FREQUENCY] if add_frequency_dim else []),
            [max(param_list ) for param_list in self.input_values.values()] + ([MAX_FREQUENCY] if add_frequency_dim else [])
        )
    
    def step_grid(self):
        """
        Generator that yields all combinations of input parameters using step grid strategy.
        This method creates a grid by stepping through each parameter's range and walking by step size.
        """
        return product(*(self.input_values[name] for name in self.input_names))
    
    def uniform_grid(self):
        """
        Generator that yields combinations of input parameters using uniform grid strategy.
        This method creates a grid by uniformly sampling each parameter's range by
        selecting values such that the total number of samples is approximately equal to num_samples.
        """

        # Calculate number of steps per parameter (nth root of n_samples)
        steps_per_param = int(np.ceil(self.n_samples ** (1 / self.n_inputs)))

        # Create uniform samples for each parameter
        param_samples = []
        for name in self.input_names:
            values = np.array(self.input_values[name])
            if len(values) < steps_per_param:
                sampled_values = values
            else:
                indices = np.linspace(0, len(values) - 1, steps_per_param, dtype=int)
                sampled_values = values[indices]
            param_samples.append(sampled_values)

        result = product(*param_samples)

        return result
        