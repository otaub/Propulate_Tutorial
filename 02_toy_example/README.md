# Propulator example

The default optimization mechanism in ``Propulate`` is that of Darwinian evolution, i.e., beneficial traits are selected,
recombined, and mutated to breed more fit individuals.
Other optimizer flavors, like particle swarm optimization (PSO), covariance matrix adaptation evolution strategy (CMA-ES),
and Nelder-Mead, are also available.
To show you how ``Propulate`` works, we use its *basic asynchronous evolutionary optimizer* to minimize two-dimensional
mathematical functions.

# Implement the script
`search.py`

## Loss function
Define a loss function that 
- receives a dictionary of parameters
- returns the sum of the square values of the parameters

## Limits
Define limits
- as dictionary
the keys are the names of the parameters used in the loss function
- the values are tuples of floats giving the lower and upper bound

## Propagator
Initialize a propagator that generates new candidate solutions based on the already evaluated ones.

## Propulator
Initialize a propulator and pass the ingredients from the previous steps.

## Run
Let the propulator propulate.

## On haicore
Adapt the job script.

# Useful links
- [https://github.com/Helmholtz-AI-Energy/propulate](https://github.com/Helmholtz-AI-Energy/propulate)
- [https://propulate.readthedocs.io/](https://propulate.readthedocs.io/)
