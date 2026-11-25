# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""Check the settings and the generated C++ code of the CPU refinement criteria.

Run from the repository root with
`PYTHONPATH=api python -m unittest discover -s api/exahype2/tests -p amr_host.py`.
"""

import tempfile
import unittest
import warnings
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


class AMRHostSettingsTest(unittest.TestCase):
    def test_prediction_mode_reports_ignored_legacy_settings(self):
        with self.assertWarnsRegex(UserWarning, "quiet_time=5.0 is ignored"):
            policy = exahype2.amr.AMR(state_scales=(1.0, 1.0), quiet_time=5.0)
        self.assertEqual(policy.quiet_time, 0.0)
        with self.assertWarnsRegex(UserWarning, "reference_time=2.0 is ignored"):
            exahype2.amr.AMR(state_scales=(1.0, 1.0), reference_time=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exahype2.amr.AMR(state_scales=(1.0, 1.0))
            exahype2.amr.AMR(state_scales=(1.0, 1.0), resolution_indicator="legacy", quiet_time=5.0, reference_time=2.0)

    def test_ignored_settings_warn_once_per_installation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = exahype2.amr.AMR(state_scales=(1.0, 1.0), reference_time=2.0)
            policy.install(solver_for("ader"))
        self.assertEqual(len([warning for warning in caught if "reference_time" in str(warning.message)]), 1)

    def test_coarsening_interval_is_validated(self):
        for invalid in (0, -1, 2**31):
            with self.assertRaises(ValueError):
                exahype2.amr.AMR(state_scales=(1.0, 1.0), coarsening_interval=invalid)
        for invalid in (True, 2.0, "2"):
            with self.assertRaises(TypeError):
                exahype2.amr.AMR(state_scales=(1.0, 1.0), coarsening_interval=invalid)
        with self.assertRaisesRegex(ValueError, "prediction mode"):
            exahype2.amr.AMR(state_scales=(1.0, 1.0), coarsening_interval=2, resolution_indicator="legacy")
        with self.assertRaisesRegex(ValueError, "execution='cpu'"):
            exahype2.amr.AMR(state_scales=(1.0, 1.0), coarsening_interval=2, execution="gpu")

    def test_parent_maximum_principle_is_validated(self):
        for invalid in (0, 1, "False", None):
            with self.assertRaisesRegex(TypeError, "parent_maximum_principle must be a boolean"):
                exahype2.amr.AMR(state_scales=(1.0, 1.0), parent_maximum_principle=invalid)
        self.assertTrue(exahype2.amr.AMR(state_scales=(1.0, 1.0)).parent_maximum_principle)

    def test_legacy_mode_without_wave_speed_reports_its_quiet_time_unit(self):
        policy = exahype2.amr.AMR(state_scales=(1.0, 1.0), resolution_indicator="legacy", quiet_time=3.0, reference_time=2.0)
        with self.assertWarnsRegex(UserWarning, "quiet_time \\* reference_time = 6.0"):
            policy.install(solver_for("fv", max_eigenvalue=False))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policy.install(solver_for("fv"))


class AMRHostGenerationTest(unittest.TestCase):
    def test_installed_policy_rejects_a_replaced_refinement_callback(self):
        for family in ("fv", "rk", "ader", "limited"):
            with self.subTest(family=family):
                solver = solver_for(family)
                exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(solver)
                with self.assertRaisesRegex(ValueError, "external_callback=True"):
                    solver.set_pde_terms(refinement_criterion="return ::exahype2::RefinementCommand::Keep;")
                with self.assertRaisesRegex(ValueError, "external_callback=True"):
                    solver.set_pde_terms(refinement_criterion=None)
                # Other terms remain open to the application.
                solver.set_pde_terms(flux="F[0] = Q[0];")

    def test_external_callback_can_forward_to_the_documented_vote(self):
        for family, member in (("fv", "_nativeCriterion"), ("limited", "_limiterCriterion")):
            with self.subTest(family=family):
                solver = solver_for(family)
                exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(solver, external_callback=True)
                solver.set_pde_terms(refinement_criterion="""
                  if (t <= 0.0 and x(0) < 0.5) return ::exahype2::RefinementCommand::Refine;
                  return {{SOLVER_NAME}}Base::amrCommand(Q, x, h, t);
                """)
                files = generate(solver)
                self.assertIn("static ::exahype2::RefinementCommand amrCommand(", files["AMRHostBase.h"])
                self.assertIn("{ return " + member + ".command(Q, x, h, t); }", files["AMRHostBase.h"])
                self.assertIn("AMRHostBase::amrCommand(Q, x, h, t)", files["AMRHost.cpph"])

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

    def test_native_criteria_bind_the_initial_mesh_before_the_initial_votes(self):
        for family, nodes in (("fv", "nullptr"), ("rk", "_amrQuadratureNodes.data()"), ("ader", "_amrQuadratureNodes.data()")):
            with self.subTest(family=family):
                solver = solver_for(family)
                exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(solver)
                base = generate(solver)["AMRHostBase.cpp"]
                begin = base.index("_nativeCriterion.beginInitialisation(")
                bind = base.index("_nativeCriterion.bindInitialState(cellConnectivity, this->downloadSolution().getView(), "
                                  + nodes + ");")
                self.assertLess(begin, bind)
                self.assertLess(bind, base.index("SolverFamily::initialise(cellConnectivity, requests, parameters);"))
        solver = solver_for("limited")
        exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(solver)
        self.assertNotIn("bindInitialState", generate(solver)["AMRHostBase.cpp"])

    def test_parent_maximum_principle_selects_the_parent_check_of_a_limited_solver(self):
        def files(**settings):
            solver = solver_for("limited")
            exahype2.amr.AMR(state_scales=(1.0, 1.0), **settings).install(solver)
            return generate(solver)

        checked = (
            "    // The proposed parent's observables must stay within the children's bounds, up to the limiter tolerance.\n"
            "    return ::exahype2::amr::parentRespectsMaximumPrinciple<Solver>(\n"
            "      parent, children, 0.0, 0.0, parameters\n"
            "    );\n")
        accepted = (
            "    // parent_maximum_principle is off: the limiter's maximum principle is not applied to the proposed parent.\n"
            "    return true;\n")
        default = files()
        self.assertEqual(default["AMRHostBase.h"].count(checked), 1)
        self.assertEqual(files(parent_maximum_principle=True), default)
        # The setting replaces the body of one generated function and changes nothing else.
        expected = dict(default)
        expected["AMRHostBase.h"] = default["AMRHostBase.h"].replace(checked, accepted)
        self.assertEqual(files(parent_maximum_principle=False), expected)

    def test_parent_maximum_principle_does_not_change_solvers_without_a_limiter(self):
        for family in ("fv", "rk", "ader"):
            with self.subTest(family=family):
                generated = []
                for enabled in (True, False):
                    solver = solver_for(family)
                    exahype2.amr.AMR(state_scales=(1.0, 1.0), parent_maximum_principle=enabled).install(solver)
                    generated.append(generate(solver))
                self.assertEqual(generated[0], generated[1])

    def test_coarsening_interval_counts_assessed_steps(self):
        solver = solver_for("ader")
        exahype2.amr.AMR(state_scales=(1.0, 1.0), coarsening_interval=4).install(solver)
        files = generate(solver)
        self.assertIn("inline static unsigned long long _nativeCriterionCoarseningAssessments=0;", files["AMRHostBase.h"])
        base = files["AMRHostBase.cpp"]
        select = base.index("_nativeCriterion.setCoarseningAssessment(_nativeCriterionCoarseningAssessments % 4 == 0);")
        self.assertLess(base.index("_nativeCriterion.beginStep("), select)
        count = base.index("if(getTimeStepping().getTimeStepSizeOfLastStep()>0.)++_nativeCriterionCoarseningAssessments;")
        self.assertLess(base.index("_nativeCriterion.finishStep();"), count)

        default = solver_for("ader")
        exahype2.amr.AMR(state_scales=(1.0, 1.0)).install(default)
        self.assertNotIn("setCoarseningAssessment", generate(default)["AMRHostBase.cpp"])

    def test_both_intervals_count_coarsening_on_assessed_steps_only(self):
        solver = solver_for("ader")
        exahype2.amr.AMR(state_scales=(1.0, 1.0), assessment_interval=3, coarsening_interval=2).install(solver)
        base = generate(solver)["AMRHostBase.cpp"]
        selected = base.index("if(_nativeCriterionAssessThisStep){\n_nativeCriterion.finishStep();")
        count = base.index("++_nativeCriterionCoarseningAssessments;")
        steps = base.index("++_nativeCriterionAcceptedSteps;")
        self.assertLess(selected, count)
        self.assertLess(count, steps)


if __name__ == "__main__":
    unittest.main()
