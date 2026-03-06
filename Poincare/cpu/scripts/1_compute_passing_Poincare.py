import time

import numpy as np
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.trajectory_helpers import PassingPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

boozmn_filename = "boozmn.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
sign_vpar = 1.0  # sign(vpar). should be +/- 1.
lam = 0.0  # lambda = v_perp^2/(v^2 B) = const. along trajectory
ntheta_poinc = 1  # Number of zeta initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_passing_map_{resolution}_{comm_size}.txt")

time1 = time.time()

bri = BoozerRadialInterpolant(
    boozmn_filename,
    order,
    no_K=True,
    comm=comm_world,
)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

poinc = PassingPoincare(
    field,
    lam,
    sign_vpar,
    mass,
    charge,
    Ekin,
    ns_poinc=ns_poinc,
    ntheta_poinc=ntheta_poinc,
    Nmaps=Nmaps,
    comm=comm_world,
    solver_options={"reltol": tol, "abstol": tol},
)
mu_per_mass = lam * Ekin / mass  # mu/m = lam * v^2/2 = lam * Ekin/mass, constant for all particles

if verbose:
    # Save initial conditions (zeta = 0 by construction)
    init_data = np.column_stack([
        poinc.s_init,
        poinc.thetas_init,
        np.zeros(len(poinc.s_init)),
        poinc.vpars_init,
        np.full(len(poinc.s_init), mu_per_mass),
    ])
    np.savetxt(
        "initial_positions.txt",
        init_data,
        header="s theta zeta vpar mu_per_mass",
    )

    # zeta = 0 at every crossing by construction
    # last 5 points are out of bounds, so skip them
    s_flat = np.concatenate([arr[:-5] for arr in poinc.s_all])
    theta_flat = np.concatenate([arr[:-5] for arr in poinc.thetas_all])
    vpar_flat = np.concatenate([arr[:-5] for arr in poinc.vpars_all])
    all_data = np.column_stack([
        s_flat,
        theta_flat,
        np.zeros(len(s_flat)),
        vpar_flat,
        np.full(len(s_flat), mu_per_mass),
    ])
    np.savetxt(
        "poincare_all_points.txt",
        all_data,
        header="s theta zeta vpar mu_per_mass",
    )

if verbose:
    poinc.plot_poincare()

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
