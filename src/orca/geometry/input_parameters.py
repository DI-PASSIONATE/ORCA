import threading
from itertools import product
from typing import Any
import numpy as np

class InputParameterIterator:
    """
    Class for keeping track of geometry input parameters.
    
    Attributes:
        input_values (dict): Dictionary mapping parameter names to their values or ranges.
    """

    def __init__(self, picking_strategy: str = "grid", frequency: list|range|np.ndarray|None = None, **input_values):
        """
        Initializes InputParameters with given input values and picking strategy.   

        Args:
            picking_strategy (str): Strategy for picking parameters ('grid', 'random', etc.).
            frequency (list|range|np.ndarray|None): Optional frequency values to include as an additional input dimension. Does not get returned by __next__ (since it's handled by palace) but is considered for min/max calculations.
        """
        # Check if all input_values are lists or ranges
        for name, values in input_values.items():
            if not isinstance(values, (list, range, np.ndarray)):
                raise ValueError(f"Input kwarg '{name}' must be a list, range, or numpy array of possible values. Found type: {type(values)} (value: {values})")

        self.picking_strategy = picking_strategy
        self.frequency = frequency
        self.n_inputs = len(input_values)
        self.input_values = input_values
        self.input_names = list(input_values.keys())
        
        self._lock = threading.Lock() # For thread-safe iteration

        # Created after set_sample_count is called
        self._iterator = None

    def set_sample_count(self, n_samples: int):
        """
        Sets the number of samples to generate. This is used for strategies
        that depend on the total number of samples, such as 'uniform_grid' and 'random'.

        Args:
            n_samples (int): Number of samples to generate.
        """
        self.n_samples = n_samples
        # Reinitialize the iterator based on the picking strategy
        if self.picking_strategy == "step_grid":
            self._iterator = self.step_grid()
        elif self.picking_strategy == "uniform_grid" or self.picking_strategy == "grid":
            self._iterator = self.uniform_grid()
        elif self.picking_strategy == "random":
            self._iterator = self.random_sampling()

    def __len__(self):
        """
        THIS RETURNS THE AMOUNT OF INPUT PARAMETERS, NOT THE AMOUNT OF GEOMETRIES THAT CAN BE GENERATED!
        """
        return len(self.input_values)        

    def __iter__(self):
        self.n_geometries_created = 0
        return self
    
    def __next__(self) -> dict[str, Any]:
        with self._lock: # Ensure thread-safe access
            self.n_geometries_created += 1 # May be used for logging or tracking
            try:
                params = next(self._iterator) # Raises StopIteration when exhausted, which is propagated to this iterator
                # Convert numpy types to Python native types to avoid type issues with downstream libraries
                params = [param.item() if isinstance(param, np.generic) else param for param in params]
                return dict(zip(self.input_names, params))
            except StopIteration:
                raise

    def get_min_max_values(self) -> tuple[list[float], list[float]]:
        """
        Returns the minimum and maximum values for each input parameter.
        Useful for normalization purposes.
        Returns:
            tuple: A tuple containing two lists - (min_values, max_values).
        """
        return (
            [min(param_list ) for param_list in self.input_values.values()] + ([min(self.frequency)] if self.frequency is not None else []),
            [max(param_list ) for param_list in self.input_values.values()] + ([max(self.frequency)] if self.frequency is not None else [])
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
        
    def random_sampling(self):
        """
        Generator that yields random combinations of input parameters.
        This method randomly samples each parameter's range for the specified number of samples.
        Given a dict of {"name": range(start, end)}, it randomly picks values from each range
        and returns a dict of {"name": value} for each sample.
        """
        for _ in range(self.n_samples):
            sampled_params = []
            for name in self.input_names:
                values = self.input_values[name]
                sampled_value = np.random.choice(values)
                sampled_params.append(sampled_value)
            yield sampled_params