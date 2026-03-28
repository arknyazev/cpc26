import sys
import time
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer_perturbed,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inputs import Inputs

if len(sys.argv) < 2:
    raise ValueError("Usage: python tracing_template.py <inputs.npz>")

inp = Inputs.from_npz(sys.argv[1])

# Load initial conditions (top nParticles rows)
ic_data = np.loadtxt(inp.ic_file, skiprows=1, max_rows=inp.nParticles)
points = np.ascontiguousarray(ic_data[:, :3])
vpar_init = np.ascontiguousarray(ic_data[:, 3])

proc0_print(f"Loaded {inp.nParticles} initial conditions from {inp.ic_file}")

# Setup logging to redirect output to file
setup_logging(f"stdout_{inp.nParticles}_{inp.resolution}_{comm_size}.txt")

## Setup radial interpolation
bri = BoozerRadialInterpolant(
    inp.boozmn_filename, 
    order=3, 
    enforce_vacuum=True, 
    comm=comm_world
)

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree=3,
    ns_interp=inp.resolution,
    ntheta_interp=inp.resolution,
    nzeta_interp=inp.resolution,
)

## Setup SAW perturbation
saw = ShearAlfvenWavesSuperposition.from_ae3d(
    eigenvector=AE3DEigenvector.load_from_numpy(filename=inp.saw_filename),
    B0=field,
    max_dB_normal_by_B0=None,
    minor_radius_meters=1.7,
)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE

# Compute magnetic moment from loaded initial conditions
vpar0 = np.sqrt(2 * Ekin / mass)
field.set_points(points)
mu_init = (vpar0**2 - vpar_init**2) / (2 * field.modB()[:, 0])

## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
time1 = time.time()
res_tys, res_zeta_hits = trace_particles_boozer_perturbed(
    saw,
    points,
    vpar_init,
    mu_init,
    mass=mass,
    charge=charge,
    comm=comm_world,
    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    forget_exact_path=True,
    abstol=inp.tol,
    reltol=inp.tol,
    tmax=inp.tmax,
)
time2 = time.time()
runtime = time2 - time1
proc0_print("Elapsed time for tracing = ", runtime)

## Post-process results to obtain lost particles
if verbose:
    from firm3d.field.trajectory_helpers import compute_loss_fraction

    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=inp.tmax)
    output_file = Path(sys.argv[1]).with_stem(Path(sys.argv[1]).stem + "_results")
    np.savez(output_file, times=times, loss_frac=loss_frac, runtime=runtime)
    proc0_print(f"Saved results to {output_file}")

    import matplotlib
    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(times, loss_frac)
    plt.xlim([1e-5, inp.tmax])
    plt.ylim([1e-3, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction of lost particles")
    plt.savefig("loss_fraction.png")
