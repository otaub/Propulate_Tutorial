import random
import propulate

def loss_fn(params):
    return params["x"] ** 2 + params["y"] ** 2

limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
rng=random.Random()
checkpoint_path="/tmp/pcheckpoints"
propulate.utils.set_logger_config()

propagator = propulate.utils.get_default_propagator(
        pop_size=10,
        limits=limits,
        rng=rng,
        )

propulator = propulate.Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=generations,
        checkpoint_path=checkpoint_path,
        )

propulator.propulate()
