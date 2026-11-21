# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""Check the settings and the generated C++ code of the CPU refinement criteria.

Run from the repository root with
`PYTHONPATH=api python -m unittest discover -s api/exahype2/tests -p amr_host.py`.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import exahype2


def solver_for(family, *, max_eigenvalue=True):
    common = dict(
        name="AMRHost", unknowns=1, auxiliary_variables=1,
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
        boundary_conditions="Qoutside[0] = Qinside[0]; Qoutside[1] = Qinside[1];",
        flux="F[0] = 0.0;",
    )
    if max_eigenvalue:
        solver.set_pde_terms(max_eigenvalue="return 1.0;")
    else:
        solver.set_pde_terms(riemann_solver="FL[0] = 0.0; FR[0] = 0.0; return 1.0;")
    return solver


def generate(solver):
    """Return the generated C++ files of a project with this solver."""
    with tempfile.TemporaryDirectory(prefix="exahype-amr-host-") as directory:
        project = exahype2.Project(
            namespace=["tests", "amr"], project_name="AMRHostGeneration",
            dimensions=2, size=[1.0, 1.0], offset=[0.0, 0.0],
            periodic_bc=[True, True], directory=directory,
        )
        project.add_solver(solver)
        with patch.object(project.output.makefile, "parse_configuration_outcome"):
            project._generate()
        generated = Path(directory) / "AMRHostGeneration.generated"
        return {path.name: path.read_text() for path in generated.iterdir() if path.is_file()}


class AMRHostGenerationTest(unittest.TestCase):
    def test_stages_precede_the_votes_of_every_host_assessment(self):
        for family, member in (("fv", "_nativeCriterion"), ("rk", "_nativeCriterion"),
                               ("ader", "_nativeCriterion"), ("limited", "_limiterCriterion")):
            with self.subTest(family=family):
                solver = solver_for(family)
                exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(solver)
                base = generate(solver)["AMRHostBase.cpp"]
                stages = base.index("if (this->getSolution().isHostAccessible()) " + member
                                    + ".assess(getTimeStepping().getTimeStamp());")
                votes = base.index("this->assessRefinement(cellConnectivity, requests, parameters);")
                self.assertLess(stages, votes)
                self.assertLess(votes, base.index(member + ".finishStep();"))


if __name__ == "__main__":
    unittest.main()
