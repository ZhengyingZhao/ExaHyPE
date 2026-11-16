// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "exahype2/dg/limiting/LimiterStatus.h"

#include <concepts>
#include <type_traits>

namespace exahype2::amr {
  /** Limiter roles used by the AMR criterion. */
  enum class CellRole { Unavailable, Regular, Halo, Limited, Troubled };

  /** Interpret the native DG subcell limiter's status array. */
  template<int SubcellCount>
  struct NativeSubcellLimiter {
    static_assert(SubcellCount > 0);

    static constexpr int SubcellsPerAxis = SubcellCount;
    static constexpr int StatusEntries = exahype2::dg::limiting::NumberOfStatusEntries;
    static constexpr int StatusIndex = static_cast<int>(exahype2::dg::limiting::StatusEntry::Status);

    static constexpr CellRole classify(const int* status) noexcept {
      if (status == nullptr) return CellRole::Unavailable;
      using NativeRole = exahype2::dg::limiting::LimiterStatus;
      switch (status[StatusIndex]) {
        case static_cast<int>(NativeRole::Regular): return CellRole::Regular;
        case static_cast<int>(NativeRole::Halo): return CellRole::Halo;
        case static_cast<int>(NativeRole::Limited): return CellRole::Limited;
        case static_cast<int>(NativeRole::Troubled): return CellRole::Troubled;
        default: return CellRole::Unavailable;
      }
    }

    static constexpr bool isKnown(CellRole role) noexcept {
      return role == CellRole::Regular or role == CellRole::Halo
          or role == CellRole::Limited or role == CellRole::Troubled;
    }

    static constexpr bool quietEligible(CellRole role) noexcept {
      return role == CellRole::Regular;
    }

    static constexpr bool requiresResolution(CellRole role) noexcept {
      return role == CellRole::Halo or role == CellRole::Limited or role == CellRole::Troubled;
    }

    static constexpr bool usesNativeFV(CellRole role) noexcept {
      return role == CellRole::Limited or role == CellRole::Troubled;
    }
  };

  /** Required status and subcell information for a limiter adapter. */
  template<typename Adapter>
  concept LimiterAdapterProtocol = requires(const int* status, CellRole role) {
    typename std::integral_constant<int, Adapter::SubcellsPerAxis>;
    requires (Adapter::SubcellsPerAxis > 0);
    { Adapter::classify(status) } -> std::same_as<CellRole>;
    { Adapter::isKnown(role) } -> std::same_as<bool>;
    { Adapter::quietEligible(role) } -> std::same_as<bool>;
    { Adapter::requiresResolution(role) } -> std::same_as<bool>;
    { Adapter::usesNativeFV(role) } -> std::same_as<bool>;
  };

  namespace detail {
    /** Use the configured limiter adapter, or the native layout by default. */
    template<typename Config, int NodesPerAxis, typename = void>
    struct LimiterAdapterFor {
      static_assert(NodesPerAxis > 0);
      using type = NativeSubcellLimiter<2 * NodesPerAxis - 1>;
    };

    template<typename Config, int NodesPerAxis>
    struct LimiterAdapterFor<Config, NodesPerAxis, std::void_t<typename Config::LimiterAdapter>> {
      static_assert(NodesPerAxis > 0);
      using type = typename Config::LimiterAdapter;
    };
  } // namespace detail
} // namespace exahype2::amr
