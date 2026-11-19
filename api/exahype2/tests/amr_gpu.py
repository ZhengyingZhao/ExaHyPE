# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""GPU-safe AMR code generation; run with PYTHONPATH=api python -m unittest discover -s api/exahype2/tests -p amr_gpu.py.

These tests render the actual solver files without requiring a GPU compiler.
The runtime RefinementVotes tests exercise the resulting host dispatch on every
configured accelerator backend.
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import exahype2


def solver_for(family):
    common = dict(
        name="AMRGPU", unknowns=1, auxiliary_variables=1,
        time_stepping=exahype2.TimeStepping.Fixed(0.01),
        precision=exahype2.Precision(storage="float", compute="double"),
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
        max_eigenvalue="return 1.0;", flux="F[0] = 0.0;",
    )
    return solver


class AMRGPUGenerationTest(unittest.TestCase):
    def render(self, family, *, external_callback=False, interval=1, sampling=False, install=True):
        solver = solver_for(family)
        solver.on_finish_time_step("// application hook before AMR installation\n")
        policy = exahype2.amr.AMR(
            state_scales=(1.0, 1.0), assessment_interval=interval,
            sample_initial_data=sampling,
        )
        if install:
            policy.install(solver, external_callback=external_callback)
        solver.on_finish_time_step("// application hook after AMR installation\n")
        with tempfile.TemporaryDirectory(prefix="exahype-amr-gpu-") as directory:
            project = exahype2.Project(
                namespace=["tests", "amr"], project_name="AMRGeneration",
                dimensions=2, size=[1.0, 1.0], offset=[0.0, 0.0],
                periodic_bc=[True, True], directory=directory,
            )
            project.add_solver(solver)
            # Generation itself is backend independent; avoid discovering an
            # unrelated installed runtime while rendering a GPU project.
            project.output.makefile.d["GPU"] = "cuda"
            with patch.object(project.output.makefile, "parse_configuration_outcome"):
                project._generate()
            generated = Path(directory) / "AMRGeneration.generated"
            files = {path.name: path.read_text() for path in generated.iterdir() if path.is_file()}
        return solver, files

    def test_all_solver_families_keep_amr_outside_device_code(self):
        for family in ("fv", "ader", "rk", "limited"):
            with self.subTest(family=family):
                solver, files = self.render(family)
                self.assertIn("RefinementCriterionOnHost = true", files["AMRGPUBase.h"])
                self.assertIn("refinementCriterion", solver.dictionary["SOLVER_HOST_PDE_TERM_DECLARATIONS"])
                self.assertNotIn("refinementCriterion", solver.dictionary["SOLVER_PDE_TERM_DECLARATIONS"])
                self.assertNotIn("refinementCriterion", solver.dictionary["SOLVER_INLINE_DEFINITIONS"])
                self.assertIn("refinementCriterion", solver.dictionary["SOLVER_HOST_INLINE_DEFINITIONS"])
                for name in ("AMRGPU.h", "AMRGPU.cpph"):
                    for block in re.findall(r"TARGET_DECLARE\b(.*?)TARGET_DECLARE_END", files[name], re.S):
                        self.assertNotIn("refinementCriterion", block)

    def test_assessment_runs_after_both_application_finish_hooks(self):
        for family in ("fv", "ader", "rk", "limited"):
            with self.subTest(family=family):
                solver, files = self.render(family)
                body = files["AMRGPUBase.cpp"]
                update = body.index("SolverFamily::performTimeStep(")
                before = body.index("// application hook before AMR installation", update)
                after = body.index("// application hook after AMR installation", update)
                vote = body.index("this->assessRefinement(", after)
                self.assertLess(update, before)
                self.assertLess(before, after)
                self.assertLess(after, vote)
                self.assertIn("downloadSolution()", solver.dictionary["START_TIME_STEP_IMPLEMENTATION"])
                self.assertNotIn("getSolution().getView()", solver.dictionary["START_TIME_STEP_IMPLEMENTATION"])
                if family == "limited":
                    for download in ("downloadStatuses()", "downloadSubcells()"):
                        self.assertIn(download, solver.dictionary["START_TIME_STEP_IMPLEMENTATION"])
                        self.assertLess(body.index(download, after), vote)

    def test_dg_assessment_and_initial_sampling_use_host_operator_tables(self):
        for family in ("ader", "rk", "limited"):
            with self.subTest(family=family):
                solver, files = self.render(family, sampling=True)
                declarations = files["AMRGPUBase.h"]
                begin = solver.dictionary["START_TIME_STEP_IMPLEMENTATION"]
                for pointer in ("Quadrature<double>::QuadraturePoints1D",
                                "DGMatrices<double>::WeightedRestrictionMatrix1D",
                                "DGMatrices<double>::ProlongationMatrix1D"):
                    self.assertNotIn(pointer, begin)
                    self.assertNotIn(pointer, declarations)
                for table in ("_amrQuadratureNodes", "_amrRestriction", "_amrProlongation"):
                    self.assertIn(table + ".data()", begin)
                    self.assertIn(table, declarations)
                self.assertIn("getQuadraturePoints1D()", declarations)
                self.assertIn("getWeightedRestrictionMatrix1D()", declarations)
                self.assertIn("getProlongationMatrix1D()", declarations)

    def test_skipped_assessments_do_not_download_limiter_mirrors(self):
        solver, _ = self.render("limited", interval=3)
        assessment = solver.dictionary["REFINEMENT_ASSESSMENT_IMPLEMENTATION"]
        self.assertRegex(assessment, r"if\s*\(\s*_limiterCriterionAssessThisStep\s*\)\s*\{")
        enabled = assessment.index("_limiterCriterionAssessThisStep")
        download = assessment.index("downloadSubcells()")
        vote = assessment.index("this->assessRefinement(")
        self.assertLess(enabled, download)
        self.assertLess(download, vote)
        self.assertIn("_limiterCriterionAssessThisStep", assessment[vote:])
        self.assertIn("getTimeStepSizeOfLastStep()>0.", assessment)

    def test_external_refinement_callback_can_remain_host_defined(self):
        solver, files = self.render("ader", external_callback=True)
        self.assertIn("refinementCriterion", solver.dictionary["SOLVER_HOST_PDE_TERM_DECLARATIONS"])
        self.assertNotIn("refinementCriterion", solver.dictionary["HOST_DEFINED_DEVICE_TERMS"])
        self.assertIn("refinementCriterion", files["AMRGPU.cpp"])

    def test_amr_does_not_change_other_solvers_device_callbacks(self):
        self.render("fv")
        solver, _ = self.render("fv", install=False)
        self.assertIn("refinementCriterion", solver.dictionary["SOLVER_PDE_TERM_DECLARATIONS"])
        self.assertNotIn("refinementCriterion", solver.dictionary["SOLVER_HOST_PDE_TERM_DECLARATIONS"])


if __name__ == "__main__":
    unittest.main()
