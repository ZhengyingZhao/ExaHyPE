// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/la/Vector.h"

namespace exahype2::terms {
  template <typename SolverType, typename Precision>
  concept HasPostprocessUpdatedPatch = requires(
    Precision* const                              Q,
    const tarch::la::Vector<DIMENSIONS, double>&  x,
    const tarch::la::Vector<DIMENSIONS, double>&  h,
    const double                                  t,
    const double                                  dt,
    const typename SolverType::RuntimeParameters& parameters
  ) {
    { SolverType::postprocessUpdatedPatch(Q, x, h, t, dt, parameters) } -> std::same_as<void>;
  };

  /**
   * A solver can define postprocessUpdatedPatch() with the number of points in Q as its second
   * argument. The subcell limiter calls this form on the subcells of every cell that it recomputes,
   * so the solver can treat the subcells like the nodes of a cell. The limiter leaves the subcells
   * of a solver without this form as the finite-volume step computes them.
   */
  template <typename SolverType, typename Precision>
  concept HasCountedPostprocessUpdatedPatch = requires(
    Precision* const                              Q,
    const int                                     numberOfPoints,
    const tarch::la::Vector<DIMENSIONS, double>&  x,
    const tarch::la::Vector<DIMENSIONS, double>&  h,
    const double                                  t,
    const double                                  dt,
    const typename SolverType::RuntimeParameters& parameters
  ) {
    { SolverType::postprocessUpdatedPatch(Q, numberOfPoints, x, h, t, dt, parameters) } -> std::same_as<void>;
  };
} // namespace exahype2::terms
