# Multiple processes for a single propulate worker

If a single evaluation is expensive, it may make sense to parallelize it.
Use thte neural network training example of exercise 04 and apply DDP.

## Job script
Have a look at the job scripts and compare them with the ones for exercise 04.

## Training script
Start from your training script from exercise 04 and adapt it for pytorch distributed data parallelism.

## Useful resources

[LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide)
[CSC ML Tutorial](https://docs.csc.fi/support/tutorials/ml-guide/)
[Getting Started with AI on LUMI workshop material](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop)
