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
    ShearAlfvenWavesSuperposition,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as CHARGE
from firm3d.util.constants import ALPHA_PARTICLE_MASS as MASS
from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
from firm3d.util.gpu_utils import boozer_saw_interpolant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inputs import Inputs

if len(sys.argv) < 2:
    raise ValueError("Usage: python tracing_template.py <inputs.npz>")

inp = Inputs.from_npz(sys.argv[1])

ic_data   = np.loadtxt(inp.ic_file, skiprows=1, max_rows=inp.nParticles)
stz_inits = np.ascontiguousarray(ic_data[:, :3])
vpar_inits = np.ascontiguousarray(ic_data[:, 3])

print(f"Loaded {inp.nParticles} initial conditions from {inp.ic_file}")

bri = BoozerRadialInterpolant(inp.boozmn_filename, 3, no_K=True)
nfp = bri.nfp
field = InterpolatedBoozerField(
    bri,
    3,
    ns_interp=inp.resolution,
    ntheta_interp=inp.resolution,
    nzeta_interp=inp.resolution,
)

saw = ShearAlfvenWavesSuperposition.from_ae3d(
    eigenvector=AE3DEigenvector.load_from_numpy(filename=inp.saw_filename),
    B0=field,
    max_dB_normal_by_B0=None,
    minor_radius_meters=1.7,
)

srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
    field, nfp, inp.resolution, inp.resolution, inp.resolution
)
print("Created GPU interpolant")

# Extract SAW harmonic data for GPU
saw_nharmonics = len(saw)
saw_omega = saw.get_wave(0).omega
s = saw.get_wave(0).phihat.get_s_basis()
saw_srange = (s[0], s[-1], len(s))
saw_m = [saw.get_wave(i).Phim for i in range(saw_nharmonics)]
saw_n = [saw.get_wave(i).Phin for i in range(saw_nharmonics)]
saw_phihats = np.ascontiguousarray(
    np.column_stack(
        [np.array([saw.get_wave(i).phihat(s_val) for s_val in s])
         for i in range(saw_nharmonics)]
    )
)
print(f"SAW: {saw_nharmonics} harmonics, omega={saw_omega:.4e}")

time1 = time.time()
last_time = firm3dpp.boozer_saw_gpu_tracing(
    quad_pts=quad_info,
    srange=srange,
    trange=trange,
    zrange=zrange,
    saw_omega=saw_omega,
    saw_srange=saw_srange,
    saw_m=saw_m,
    saw_n=saw_n,
    saw_phihats=saw_phihats,
    saw_nharmonics=saw_nharmonics,
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
