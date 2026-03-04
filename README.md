# job-assist

Interactive SLURM sbatch script generator for HPC clusters.

Run `job-assist` on any SLURM cluster and it will auto-detect partitions, GRES/TRES, accounts, QOS, GPU types, and node features — then walk you through building a robust submission script.

## Features

- **Cluster auto-detection** — queries `scontrol`, `sinfo`, and `sacctmgr` to discover available resources
- **GPU-aware** — detects GRES GPU types (A100, V100, etc.) and offers them as choices
- **Account & QOS support** — finds your SLURM accounts and associated QOS policies
- **Node feature constraints** — surfaces node features for `--constraint` selection
- **Validated input** — wall time, memory, and resource values are validated as you type
- **Script preview** — shows the complete script before saving
- **Cluster-agnostic** — works on any SLURM cluster (Rocky 8/9/10+, any config)

## Quick Start

```bash
# Install into a virtualenv or prefix
pip install /path/to/job-assist

# Run it
job-assist

# Just show detected cluster info
job-assist --detect-only
```

## Installation as Lmod Module

```bash
# Build and install to a prefix
pip install . --prefix=/apps/job-assist/0.1.0

# Copy the modulefile
cp modulefile/job-assist.lua /apps/modulefiles/job-assist/0.1.0.lua

# Edit the base_dir in the .lua file to match your prefix
# Users can then:
module load job-assist
job-assist
```

## Usage

```
usage: job-assist [-h] [--version] [--detect-only] [-o OUTPUT] [--no-detect]

Interactive SLURM sbatch script generator

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --detect-only         print detected cluster info and exit
  -o OUTPUT, --output OUTPUT
                        output file path (default: <job_name>.sbatch)
  --no-detect           skip cluster auto-detection (manual entry)
```

## Requirements

- Python >= 3.8
- SLURM (for auto-detection; works without it in manual mode)

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```
