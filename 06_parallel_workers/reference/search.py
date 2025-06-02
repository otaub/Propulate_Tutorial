import pathlib
import random

import numpy as np
from mpi4py import MPI

from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config


def loss_fn(params, comm = MPI.COMM_SELF) -> float:
    """
    Parallel sphere function to showcase using multi-rank workers in Propulate.

    Sphere function: continuous, convex, separable, differentiable, unimodal

    Input domain: -5.12 <= x, y <= 5.12
    Global minimum 0 at (x, y) = (0, 0)

    Parameters
    ----------
    params : Dict[str, float]
        The function parameters.
    comm : MPI.Comm
        The communicator of the worker.

    Returns
    -------
    float
        The function value.
    """
    assert comm != MPI.COMM_SELF
    term = list(params.values())[comm.rank] ** 2  # Each rank squares one of the inputs.
    return comm.allreduce(term)  # Return the sum over all squared inputs.


limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
seed = 42
rng = random.Random(seed + MPI.COMM_WORLD.rank)
checkpoint_path = "/tmp/pcheckpoints"
set_logger_config()
pollination = False
num_islands = 2
migration_prob = 0.3
ranks_per_worker=2

if __name__ == "__main__":
    full_world_comm = MPI.COMM_WORLD  # Get full world communicator.

    # Set up separate logger for Propulate optimization.
    set_logger_config()

    rng = random.Random(seed + full_world_comm.rank)  # Separate random number generator for optimization.
    # Set callable function + search-space limits.

    # Set up evolutionary operator.
    propagator = get_default_propagator(pop_size=10, limits=limits, rng=rng)

    # Set up island model.
    islands = Islands(
        loss_fn=loss_fn,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=generations,  # Overall number of generations
        num_islands=num_islands,  # Number of islands
        migration_probability=migration_prob,  # Migration probability
        pollination=pollination,  # Whether to use pollination or migration
        checkpoint_path=checkpoint_path,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS ----
        ranks_per_worker=ranks_per_worker,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.propulate()
