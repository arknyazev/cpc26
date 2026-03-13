#!/usr/bin/env python
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.gpu_utils import boozer_interpolant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inputs import Inputs

if len(sys.argv) < 2:
    raise ValueError("Usage: python reference_script.py <inputs.npz>")

inp = Inputs.from_npz(sys.argv[1])

# Load initial conditions (top nParticles rows)
ic_data = np.loadtxt(inp.ic_file, skiprows=1, max_rows=inp.nParticles)
stz_inits = ic_data[:, :3]
vpar_inits = ic_data[:, 3]

print(f"Loaded {inp.nParticles} initial conditions from {inp.ic_file}")

# Build GPU interpolant
bri = BoozerRadialInterpolant(inp.boozmn_filename, 3)
nfp = bri.nfp
field = InterpolatedBoozerField(
    bri,
    3,
    ns_interp=inp.resolution,
    ntheta_interp=inp.resolution,
    nzeta_interp=inp.resolution,
)
srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, inp.resolution, inp.resolution, inp.resolution)
print("Created GPU interpolant")

# Trace on GPU
time1 = time.time()
last_time = firm3dpp.boozer_gpu_tracing(
    quad_pts=quad_info,
    srange=srange,
    trange=trange,
    zrange=zrange,
    stz_init=stz_inits,
    m=MASS,
    q=CHARGE,
    vtotal=sqrt(2 * ENERGY / MASS),
    vtang=vpar_inits,
    tmax=inp.tmax,
    tol=inp.tol,
    psi0=field.psi0,
    nparticles=inp.nParticles,
)
time2 = time.time()
runtime = time2 - time1
print(f"Elapsed time for tracing = {runtime}")

last_time = np.reshape(last_time, (inp.nParticles, 5))
loss_frac = np.mean(last_time[:, 0] < inp.tmax)

output_file = Path(sys.argv[1]).with_stem(Path(sys.argv[1]).stem + "_results")
np.savez(output_file, loss_frac=loss_frac, runtime=runtime)
print(f"Saved results to {output_file}")
