import sys
import time
from pathlib import Path
import numpy as np
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

if len(sys.argv) < 2:
    raise ValueError("Usage: python sample_fusion_distribution.py <boozmn_filename>/<wout_filename>")

boozmn_filename = sys.argv[1]
device_stem = Path(boozmn_filename).stem

time1 = time.time()

resolution = 48  # Resolution for field interpolation
nParticles = 80000  # Number of particles to sample
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")

## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized
# stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: (1 - s**5)  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))

points = initialize_position_profile(field, nParticles, reactivity, comm=comm_world)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)
vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)

proc0_print(f"IC generation time: {time.time() - time1:.2f}s")

# Save initial conditions
output_dir = Path(__file__).parent / device_stem
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "initial_conditions.txt"

data = np.column_stack([points, vpar_init])
np.savetxt(
    output_file,
    data,
    header="s_init theta_init zeta_init vpar_init",
)
proc0_print(f"Saved {nParticles} initial conditions to {output_file}")
