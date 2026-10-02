"""One seed for every random number generator of a pipeline run."""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """
    Seed the global random number generators of Python, NumPy and, if installed, PyTorch.

    Libraries that keep generators of their own (the input parameter sampler,
    scikit-learn's splits, Optuna's sampler) are not reached this way; the stages hand
    them :attr:`~orca.pipeline.context.PipelineContext.seed` explicitly.

    Args:
        seed (int): The seed of the run.
    """
    random.seed(seed)
    # The legacy global generator is what geometry code calling np.random.* draws from
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch  # optional: only installed with ORCA's "train" extra
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)  # seeds every CUDA device too
