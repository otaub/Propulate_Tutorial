# Getting Started

## Set up: LUMI

### Log in and prepare workspace
- [ ] `ssh <user>@lumi.csc.fi`
- [ ] `cd /scratch/project_465001989`
- [ ] `mkdir $USER`
- [ ] `cd $USER`

### Download course material and software
- [ ] `git clone https://github.com/otaub/Propulate_Tutorial.git`
- [ ] `git clone https://github.com/Helmholtz-AI-Energy/propulate.git`

### Set up virtual environment
- [ ] `module use /appl/local/csc/modulefiles/`
- [ ] `module load pytorch/2.5`
- [ ] `python -m venv pvenv --system-site-packages`
- [ ] `source pvenv/bin/activate`
- [ ] `cd propulate`
- [ ] `pip install -e .`

## Set up: HAICORE
