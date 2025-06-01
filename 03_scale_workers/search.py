import random
from mpi4py import MPI
import propulate

def loss_fn(params):
    return params["x"] ** 2 + params["y"] ** 2

limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
num_islands = 2
seed = 42
rng=random.Random(seed + MPI.COMM_WORLD.rank)
checkpoint_path="/tmp/pcheckpoints"
propulate.utils.set_logger_config()
pollination=False

propagator = propulate.utils.get_default_propagator(
        pop_size=200,
        limits=limits,
        rng=rng,
        )

islands = propulate.Islands(
    loss_fn=loss_fn,
    propagator=propagator,
    rng=rng,
    generations=generations,
    num_islands=num_islands,
    migration_probability=0.1,
    pollination=pollination,
    checkpoint_path=checkpoint_path,
)

islands.propulate()
# islands.summarize()
