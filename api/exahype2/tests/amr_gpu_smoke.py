# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""Generate an AMR integration smoke test with the selected CPU or GPU runtime.

Set PEANO_BUILD_DIR, then run this script with --build-mode Asserts and
--dimensions 2 or 3. Run the resulting ExaHyPE executable with -ns 0.
"""

import argparse
from pathlib import Path

import exahype2


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="build/amr-gpu-smoke")
    arguments, _ = parser.parse_known_args()
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = exahype2.Project(
        namespace=["tests", "amr"], project_name="AMRGPUSmoke", dimensions=2,
        size=[1.0] * 3, offset=[0.0] * 3, periodic_bc=[True] * 3,
        min_end_time=0.03, max_end_time=0.03, min_grid_levels=1,
        max_grid_levels=2, directory=str(output),
    )
    for family in ("fv", "ader", "rk", "limited"):
        common = dict(
            name="AMR" + family.title(), unknowns=1, auxiliary_variables=1,
            time_stepping=exahype2.TimeStepping.Fixed(0.01),
        )
        if family == "fv":
            solver = exahype2.fv.Solver(patch_size=2, scheme=exahype2.fv.Godunov(), **common)
        elif family == "rk":
            solver = exahype2.dg.rk.Solver(rk_order=2, dg_order=1, **common)
        else:
            solver = exahype2.dg.ader.Solver(order=1, **common)
            if family == "limited":
                solver.add_subcell_limiter(exahype2.fv.Godunov(), number_of_dmp_observables=1)
                solver.set_pde_terms(
                    physical_admissibility="return true;",
                    dmp_observables="observables[0] = Q[0];",
                )
        solver.set_pde_terms(
            initial_condition="Q[0] = 1.0; Q[1] = 0.0;",
            max_eigenvalue="return 1.0;", flux="F[0] = 0.0;",
        )
        # The kernel changes the initially constant accepted state. The AMR
        # assessment must observe it after the completed step on either backend.
        solver.postprocess_updated_patch = """
if (t > 0.0) for (int point = 0; point < PointsPerCell; point++) {
  Q[point * NumberOfData] = 1.0 + 0.1 * (point % 2);
}
"""
        exahype2.amr.AMR(
            state_scales=(1.0, 1.0), transfer_tolerance=0.001,
            sample_initial_data=family != "fv",
        ).install(solver)
        if family != "limited":
            # Accumulate evidence: another solver may take additional steps
            # after this one reaches its terminal time and clears its caches.
            solver.add_declarations("inline static bool _sawAMRDetail = false;\n")
            solver._refinement_assessment_implementation += (
                "_sawAMRDetail = _sawAMRDetail or getAMRStatistics().refineEvidence > 0;\n"
            )
            solver._finish_simulation_implementation = (
                'if (not _sawAMRDetail) throw std::runtime_error("Missing accepted AMR evidence");\n'
                + solver._finish_simulation_implementation
            )
        project.add_solver(solver)
    project.generate_and_build()


if __name__ == "__main__":
    main()
