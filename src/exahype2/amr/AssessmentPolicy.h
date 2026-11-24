// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <type_traits>

namespace exahype2::amr {
  /**
   * The checks of the criterion for a solver without a limiter. The flags mean:
   *  - NativeArithmetic: sums, products and norms use the overflow-checked helpers of NativeNeighbours.h.
   *  - RequireSingleRank: the criterion refuses to run on more than one MPI rank.
   *  - CheckNeighbours: the score of a cell includes its faces.
   *  - CheckQuietDetail: in legacy mode only a resolved cell collects quiet time.
   *  - CheckCommandPhysical: a cell whose solution is not physically admissible refines.
   *  - UseLocalConstancy: a cell that the limiter marks keeps its size where the solution is constant.
   * The texts are the messages of the exceptions that the criterion throws.
   */
  struct NativeAssessmentPolicy {
    static constexpr bool NativeArithmetic = true;
    static constexpr bool RequireSingleRank = true;
    static constexpr bool CheckNeighbours = true;
    static constexpr bool CheckQuietDetail = true;
    static constexpr bool CheckCommandPhysical = true;
    static constexpr bool UseLocalConstancy = false;
    static constexpr const char* InvalidUnits = "Independent AMR criterion requires finite, positive transfer/time units.";
    static constexpr const char* InvalidScale = "Independent AMR criterion requires an explicit positive scale for every state component.";
    static constexpr const char* MissingNodes = "DG AMR requires the solver's original quadrature nodes.";
    static constexpr const char* InvalidNodes = "DG AMR received invalid quadrature nodes.";
    static constexpr const char* InvalidReferenceTime = "Independent AMR criterion reference time must be finite and positive.";
  };

  /** The checks of the criterion for ADER-DG with the subcell limiter. NativeAssessmentPolicy explains the flags. */
  struct LimiterCompatibilityPolicy {
    static constexpr bool NativeArithmetic = false;
    static constexpr bool RequireSingleRank = false;
    static constexpr bool CheckNeighbours = true;
    static constexpr bool CheckQuietDetail = true;
    static constexpr bool CheckCommandPhysical = false;
    static constexpr bool UseLocalConstancy = true;
    static constexpr const char* InvalidUnits = "Limiter criterion requires finite, positive transfer/time units.";
    static constexpr const char* InvalidScale = "Limiter criterion requires an explicit positive scale for every state component.";
    static constexpr const char* MissingNodes = "Limiter criterion requires the solver's original quadrature nodes.";
    static constexpr const char* InvalidNodes = "Limiter criterion received invalid quadrature nodes.";
    static constexpr const char* InvalidReferenceTime = "Limiter criterion reference time must be finite and positive.";
  };

  /** The checks that fit the Accepted type: those for the limiter if the solver has subcells, the others if not. */
  template<typename Accepted>
  using DefaultAssessmentPolicy = std::conditional_t<Accepted::HasHybridSubcells,
    LimiterCompatibilityPolicy, NativeAssessmentPolicy>;
} // namespace exahype2::amr
