# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""Build a small four-solver problem that exercises the CPU or GPU refinement criterion.

Set PEANO_BUILD_DIR, then run this script with --build-mode Asserts and
--dimensions 2 or 3. Run the resulting ExaHyPE executable with -ns 0.
"""

import argparse
from pathlib import Path

import exahype2


def add_limited_accepted_state_probe(solver, execution):
    """Check that the limiter's FV subcells can trigger refinement while the DG values stay constant."""
    solver.add_declarations("""
  inline static bool _sawLimitedFVKeep = false;
  inline static bool _sawLimitedFVRefinement = false;
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  struct InjectAcceptedFVDetail final {
    ::tarch::Span<const int> cells;
    typename ::tarch::accelerator::ChunkedMemoryPool<StoragePrecision>::ConstView solution, subcells;
    typename ::tarch::accelerator::ChunkedMemoryPool<int>::View statuses;
    bool hasDetail;
    GPU_CALLABLE_METHOD void operator()(int entry) const {
      const int slot = cells[entry];
      auto* nodes = const_cast<StoragePrecision*>(solution.getChunk(slot));
      auto* fv = const_cast<StoragePrecision*>(subcells.getChunk(slot));
      auto* status = statuses.getChunk(slot);
      // Only the FV subcells carry the alternating signal.
      for (int point = 0; point < Solver::PointsPerCell; ++point) {
        nodes[point * Solver::NumberOfData] = 1.0;
        nodes[point * Solver::NumberOfData + 1] = 0.0;
      }
      for (int point = 0; point < ::peano4::power(Solver::PatchSize, DIMENSIONS); ++point) {
        fv[point * Solver::NumberOfData] = 1.0 + (hasDetail ? 0.1 : 0.0) * (point % 2);
        fv[point * Solver::NumberOfData + 1] = 0.0;
      }
      using namespace ::exahype2::dg::limiting;
      status[static_cast<int>(StatusEntry::Status)] = static_cast<int>(LimiterStatus::Troubled);
      status[static_cast<int>(StatusEntry::PreviousStatus)] = static_cast<int>(LimiterStatus::Troubled);
      status[static_cast<int>(StatusEntry::WasTroubled)] = 1;
      status[static_cast<int>(StatusEntry::PredictorDiverged)] = 0;
    }
  };
""")
    solver.on_finish_time_step("""
if (getTimeStepping().getTimeStepSizeOfLastStep() > 0.0) {
  auto& limiter = const_cast<typename SolverFamily::LimiterType&>(getLimiter());
  const auto cells = cellConnectivity.onDevice().getCells();
  ::tarch::accelerator::parallelFor(::tarch::accelerator::Range(cells.size),
    InjectAcceptedFVDetail<>{cells, getSolution().getView(), getLimiter().getSubcells(), limiter.getStatuses(),
      getTimeStepping().getTimeStamp() > 0.015});
}
""")
    # A CPU query needs the cell state. A GPU query reads the cell's saved vote.
    state = "static_cast<const StoragePrecision*>(nullptr)"
    mirror = ""
    if execution == "cpu":
        mirror = "const auto accepted = this->downloadSolution().getView();\n"
        state = "accepted.getChunk(slot)"
    probe = """
if (getTimeStepping().getTimeStepSizeOfLastStep() > 0.0) {
  %(mirror)s
  for (int slot : cellConnectivity.getCells()) {
    const auto width = cellConnectivity.getCellWidth(slot);
    const auto command = _limiterCriterion.command(%(state)s, cellConnectivity.getCellCentre(slot), width,
      getTimeStepping().getTimeStamp());
    if (getTimeStepping().getTimeStamp() < 0.015 and command == ::exahype2::RefinementCommand::Keep) {
      _sawLimitedFVKeep = true;
    }
    if (getTimeStepping().getTimeStamp() > 0.015
        and ::tarch::la::greater(::tarch::la::max(width), getTimeStepping().getLimits().minAdmissiblePatchSize)
        and command == ::exahype2::RefinementCommand::Refine) {
      _sawLimitedFVRefinement = true;
    }
  }
}
""" % {"mirror": mirror, "state": state}
    finish = "_limiterCriterion.finishStep();"
    if solver._refinement_assessment_implementation.count(finish) != 1:
        raise RuntimeError("The limited smoke probe requires one AMR finishStep hook.")
    solver._refinement_assessment_implementation = solver._refinement_assessment_implementation.replace(
        finish, probe + finish
    )
    solver._finish_simulation_implementation = (
        'if (not _sawLimitedFVKeep) throw std::runtime_error("Missing accepted constant native FV AMR Keep");\n'
        'if (not _sawLimitedFVRefinement) throw std::runtime_error("Missing accepted native FV AMR refinement");\n'
        + solver._finish_simulation_implementation
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="build/amr-gpu-smoke")
    parser.add_argument("--criterion-execution", choices=("cpu", "gpu"), default="cpu",
                        help="Place the complete AMR assessment on the CPU or configured accelerator.")
    parser.add_argument("--storage-precision", choices=("float", "double"), default="double",
                        help="Stored state precision; computations use double precision.")
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
            precision=exahype2.Precision(storage=arguments.storage_precision, compute="double"),
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
        # Alternating values create a refinement signal after the initial step.
        solver.postprocess_updated_patch = """
if (t > 0.0) for (int point = 0; point < PointsPerCell; point++) {
  Q[point * NumberOfData] = 1.0 + 0.1 * (point % 2);
}
"""
        exahype2.amr.AMR(
            state_scales=(1.0, 1.0), transfer_tolerance=0.001,
            sample_initial_data=family != "fv",
            execution=arguments.criterion_execution,
        ).install(solver)
        if family == "limited":
            add_limited_accepted_state_probe(solver, arguments.criterion_execution)
        else:
            # Each solver can finish at a different time, so remember the signal across steps.
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
