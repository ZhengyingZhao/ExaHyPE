// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "exahype2/dg/limiting/LimiterStatus.h"

#include <concepts>
#include <type_traits>

TARGET_DECLARE
namespace exahype2::amr {
  /** The limiter status of a cell as the criterion sees it. Troubled and Limited cells compute on FV subcells,
   * Halo cells surround them, Regular cells are ordinary DG cells. Unavailable: no valid status is stored. */
  enum class CellRole { Unavailable, Regular, Halo, Limited, Troubled };

  /** Reads the status that the subcell limiter of ADER-DG stores for a cell and answers what the criterion
   * asks about it. SubcellCount is the number of subcells per axis. */
  template<int SubcellCount>
  struct NativeSubcellLimiter {
    static_assert(SubcellCount > 0);

    static constexpr int SubcellsPerAxis = SubcellCount;
    static constexpr int StatusEntries = exahype2::dg::limiting::NumberOfStatusEntries;
    static constexpr int StatusIndex = static_cast<int>(exahype2::dg::limiting::StatusEntry::Status);

    /** Returns the role of a cell from its stored status. */
    static GPU_CALLABLE_METHOD constexpr CellRole classify(const int* status) noexcept {
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

    /** Is the role one of the four that the limiter sets? */
    static GPU_CALLABLE_METHOD constexpr bool isKnown(CellRole role) noexcept {
      return role == CellRole::Regular or role == CellRole::Halo
          or role == CellRole::Limited or role == CellRole::Troubled;
    }

    /** May the cell coarsen? Only a regular cell may. */
    static GPU_CALLABLE_METHOD constexpr bool quietEligible(CellRole role) noexcept {
      return role == CellRole::Regular;
    }

    /** Does the cell belong to a limited region: troubled, limited or halo? */
    static GPU_CALLABLE_METHOD constexpr bool requiresResolution(CellRole role) noexcept {
      return role == CellRole::Halo or role == CellRole::Limited or role == CellRole::Troubled;
    }

    /** Does the cell hold its solution as FV subcell values? */
    static GPU_CALLABLE_METHOD constexpr bool usesNativeFV(CellRole role) noexcept {
      return role == CellRole::Limited or role == CellRole::Troubled;
    }
  };

  /** What a limiter adapter has to provide: the number of subcells per axis and the five functions of
   * NativeSubcellLimiter. */
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
    /** The limiter adapter that Config names, or NativeSubcellLimiter with 2N-1 subcells per axis if Config
     * names none. */
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
TARGET_DECLARE_END
