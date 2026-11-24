# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import argparse

import exahype2

# The two forms of the hook that the solver runs on a cell after its update. Both turn water at or
# below hThreshold into dry ground. The solver calls the first form on the nodes of a cell. The
# subcell limiter calls the second form on the subcells of a cell that it recomputes.
DRY_STATE_CLEANUP = """
  static INLINE_METHOD GPU_CALLABLE_METHOD void postprocessUpdatedPatch(
    double* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    double t,
    double dt,
    const RuntimeParameters& parameters
  ) {
    postprocessUpdatedPatch(Q, PointsPerCell, x, h, t, dt, parameters);
  }

  static INLINE_METHOD GPU_CALLABLE_METHOD void postprocessUpdatedPatch(
    double* Q,
    int numberOfPoints,
    [[maybe_unused]] const tarch::la::Vector<DIMENSIONS, double>& x,
    [[maybe_unused]] const tarch::la::Vector<DIMENSIONS, double>& h,
    [[maybe_unused]] double t,
    [[maybe_unused]] double dt,
    [[maybe_unused]] const RuntimeParameters& parameters
  ) {
    applications::exahype2::exaskred::limiting::resetDryStates<
      Shortcuts,
      NumberOfUnknowns,
      NumberOfAuxiliaryVariables
    >(Q, numberOfPoints, hThreshold);
  }
"""


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="show this help message and continue with project help",
    )
    parser.add_argument("-et", "--end-time", type=float, default=10.0)
    parser.add_argument("--degrees-of-freedom", type=int, default=16)
    parser.add_argument("-l", "--grid-levels", type=int, default=None)
    parser.add_argument("-lmin", "--min-grid-levels", type=int, default=5)
    parser.add_argument("-lmax", "--max-grid-levels", type=int, default=5)
    parser.add_argument(
        "--dry-state-cleanup",
        action="store_true",
        help="turn water at or below the dry threshold into dry ground after every update",
    )
    args, _ = parser.parse_known_args()
    if args.grid_levels is not None:
        args.min_grid_levels = args.grid_levels
        args.max_grid_levels = args.grid_levels
    if args.help:
        parser.print_help()
        print()
    return args


args = parse_arguments()
dg_order = max(1, (args.degrees_of_freedom - 1) // 2)

project = exahype2.Project(
    namespace=["applications", "exahype2", "exaskred"],
    project_name="OscillatingLake",
    dimensions=2,
    offset=[-2.0, -2.0],
    size=[4.0, 4.0],
    periodic_bc=[False, False],
    executable="ExaSkred-Tsunami",
    min_end_time=args.end_time,
    max_end_time=args.end_time,
    min_grid_levels=args.min_grid_levels,
    max_grid_levels=args.max_grid_levels,
)
if project.dimensions != 2:
    raise ValueError("The oscillating-lake application is two-dimensional.")

aderdg_solver = exahype2.dg.ader.Solver(
    name="ADERDGSolver",
    order=dg_order,
    unknowns={"h": 1, "hu": project.dimensions, "b": 1},
    auxiliary_variables=0,
    time_stepping=exahype2.TimeStepping.Adaptive(0.5),
    basis=exahype2.dg.GaussLobatto,
    linear=False,
)

aderdg_solver.set_pde_terms(
    initial_condition=True,
    boundary_conditions=True,
    refinement_criterion=True,
    flux=True,
    max_eigenvalue=True,
    ncp=True,
    riemann_solver=True,
)

aderdg_solver.set_pde_terms(
    subcell_riemann_solver=True,
    physical_admissibility=True,
)
aderdg_solver.add_subcell_limiter(exahype2.fv.Godunov())

# A retreating shoreline leaves water at or below hThreshold behind. The admissibility test
# accepts a cell as dry only when it holds no water, so such a cell stays on the subcell limiter.
# The cleanup removes this water and the cell returns to the ADER-DG scheme.
if args.dry_state_cleanup:
    aderdg_solver.add_includes("applications/exaskred/Limiting.h")
    aderdg_solver.add_solver_declarations(DRY_STATE_CLEANUP)

aderdg_solver.set_pde_terms(analytical_solution=True)
aderdg_solver.measure_error(every=args.end_time / 100.0)

project.add_solver(aderdg_solver)

project.add_constant("g", 9.81)
project.add_constant("hThreshold", 1e-5)
project.add_constant("LakeRadius", 1.0)
project.add_constant("ReferenceDepth", 0.1)
project.add_constant("OrbitDisplacement", 0.5)

project.output.makefile.add_include_directory("../../../..")
project.output.makefile.add_h_file("ADERDGSolver.cpph")
project.generate_and_build()
