import random

from mpi4py import MPI

from propulate import Propulator
from propulate.propagators import ActiveCMA, CMAPropagator
from propulate.utils import set_logger_config

def loss_fn(params):
    return params["x"] ** 2 + params["y"] ** 2

limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
seed = 42
rng=random.Random(seed + MPI.COMM_WORLD.rank)
checkpoint_path="/tmp/pcheckpoints"
set_logger_config()
pollination=False


if __name__ == "__main__":

    rng = random.Random(seed + MPI.COMM_WORLD.rank)
    adapter = ActiveCMA()

    propagator = CMAPropagator(adapter, limits, rng=rng)

    # Set up propulator performing actual optimization.
    propulator = Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=generations,
        checkpoint_path=checkpoint_path,
    )

    # Run optimization and print summary of results.
    propulator.propulate()
