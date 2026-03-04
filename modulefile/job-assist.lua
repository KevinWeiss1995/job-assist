-- -*- lua -*-
-- Lmod modulefile for job-assist
-- Install instructions:
--   1. pip install /path/to/job-assist --prefix=/apps/job-assist/0.1.0
--   2. Place this file at $MODULEPATH/job-assist/0.1.0.lua
--   3. Adjust base_dir below to match your install prefix

local version = "0.1.0"
local base_dir = "/apps/job-assist/" .. version

whatis("Name:        job-assist")
whatis("Version:     " .. version)
whatis("Description: Interactive SLURM sbatch script generator for HPC clusters")
whatis("URL:         https://github.com/your-org/job-assist")

help([[
job-assist - Interactive SLURM sbatch script generator

Usage:
  $ job-assist              # interactive mode
  $ job-assist --detect-only  # show cluster info

Run 'job-assist --help' for all options.
]])

-- Adjust the Python version directory to match what's installed
local python_lib = pathJoin(base_dir, "lib", "python3.*/site-packages")
-- Use a glob-safe approach: find the actual python dir
local python_dirs = {
    pathJoin(base_dir, "lib", "python3.8",  "site-packages"),
    pathJoin(base_dir, "lib", "python3.9",  "site-packages"),
    pathJoin(base_dir, "lib", "python3.10", "site-packages"),
    pathJoin(base_dir, "lib", "python3.11", "site-packages"),
    pathJoin(base_dir, "lib", "python3.12", "site-packages"),
    pathJoin(base_dir, "lib", "python3.13", "site-packages"),
}

prepend_path("PATH", pathJoin(base_dir, "bin"))

for _, d in ipairs(python_dirs) do
    if isDir(d) then
        prepend_path("PYTHONPATH", d)
        break
    end
end
