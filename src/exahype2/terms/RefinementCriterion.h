// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "exahype2/RefinementCommand.h"
#include "tarch/Span.h"
#include "tarch/la/Vector.h"

namespace peano4 {
  class CellConnectivity;
}

namespace exahype2 {
  class RefinementRequests;
  class TimeStepping;
}

namespace exahype2::terms {
  /** True when the solver asks to evaluate its refinement criterion on the CPU. */
  template <typename SolverType>
  inline constexpr bool RefinementCriterionOnHost = [] {
    if constexpr (requires { SolverType::RefinementCriterionOnHost; }) {
      return static_cast<bool>(SolverType::RefinementCriterionOnHost);
    } else {
      return false;
    }
  }();

  template <typename SolverType, typename Precision>
  concept HasRefinementCriterion = requires(
    const Precision* const                        Q,
    const tarch::la::Vector<DIMENSIONS, double>&  x,
    const tarch::la::Vector<DIMENSIONS, double>&  h,
    const double                                  t,
    const typename SolverType::RuntimeParameters& parameters
  ) {
    { SolverType::refinementCriterion(Q, x, h, t, parameters) } -> std::same_as<exahype2::RefinementCommand>;
  };

  /** True when the solver can decide on the device which cells should refine or coarsen. */
  template <typename SolverType, typename SolutionPool, typename RuntimeParameters>
  concept HasDeviceAMRAssessment = requires(
    SolverType&                     solver,
    const peano4::CellConnectivity& mesh,
    const SolutionPool&             solution,
    TimeStepping&                   clock,
    RefinementRequests&             requests,
    const RuntimeParameters&        parameters
  ) {
    solver.assessAMROnDevice(mesh, solution.getView(), clock, requests, parameters, tarch::Span<const int>{}, true);
  };
} // namespace exahype2::terms
