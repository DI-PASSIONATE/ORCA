"""Neural-network architectures for the surrogate models.

Importing the built-in models registers them under their short names
(``"mlp"``, ...), so ``ModelTrainer(model="mlp")`` resolves without the caller
importing the class itself. User models register through ``register_model``.
"""

from orca.training.models import mlp  # noqa: F401 - registration side effect
