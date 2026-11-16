// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <type_traits>

namespace exahype2::amr {
  /** Choose numerical checks independently of limiter status. */
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

  /** Assess accepted DG/FV states while retaining limiter-specific protections. */
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

  /** Select the assessment policy from the accepted-state layout. */
  template<typename Accepted>
  using DefaultAssessmentPolicy = std::conditional_t<Accepted::HasHybridSubcells,
    LimiterCompatibilityPolicy, NativeAssessmentPolicy>;
} // namespace exahype2::amr
