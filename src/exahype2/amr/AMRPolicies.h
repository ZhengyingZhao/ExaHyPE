// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "AssessmentPolicy.h"
#include "AMRMath.h"
#include "tarch/accelerator/ChunkedMemoryPool.h"

namespace exahype2::amr {
  /** Tag for AMR core implementations. */
  struct AMRCoreTag {};

  enum class AcceptedRepresentation { DGNodal, FVCellAverages };

  /** Read-only accepted cell values and their layout. */
  template<typename P>
  struct AcceptedCellView {
    const P* values{nullptr};
    int pointsPerAxis{0};
    AcceptedRepresentation representation{AcceptedRepresentation::DGNodal};
    [[nodiscard]] int numberOfStates() const { return peano4::power(pointsPerAxis, DIMENSIONS); }
    [[nodiscard]] bool isFiniteVolume() const { return representation == AcceptedRepresentation::FVCellAverages; }
  };

  struct NoStatusView {
    [[nodiscard]] const int* getChunk(int) const { return nullptr; }
  };

  /** Read standalone FV means or DG nodal values. */
  template<int N, int V, typename C, typename P>
  struct NativeAcceptedSolutionAdapter {
    using Subcells = typename tarch::accelerator::ChunkedMemoryPool<P>::ConstView;
    static constexpr int SubcellsPerAxis = N;
    static constexpr bool HasHybridSubcells = false;
    static AcceptedCellView<P> accepted(int, const P* state, const int*, Subcells) {
      return {state, N, C::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal};
    }
  };

  /** Read DG values or the limiter's accepted FV subcells. */
  template<int N, int V, typename P, typename Layout>
  struct HybridAcceptedSolutionAdapter {
    using LimiterAdapter = Layout;
    using Subcells = typename tarch::accelerator::ChunkedMemoryPool<P>::ConstView;
    static constexpr int SubcellsPerAxis = Layout::SubcellsPerAxis;
    static constexpr bool HasHybridSubcells = true;
    static AcceptedCellView<P> accepted(int slot, const P* state, const int* status, Subcells subcells) {
      const auto role = Layout::classify(status);
      if (not Layout::isKnown(role)) return {};
      if (not Layout::usesNativeFV(role)) return {state, N, AcceptedRepresentation::DGNodal};
      constexpr int values = peano4::power(SubcellsPerAxis, DIMENSIONS) * V;
      if (subcells.segments == nullptr or subcells.paddedChunkSize < values) return {};
      return {subcells.getChunk(slot), SubcellsPerAxis, AcceptedRepresentation::FVCellAverages};
    }
  };

  /** No limiter-specific restrictions on AMR decisions. */
  struct NoLimiterEvidence {
    static bool available(const int*) { return true; }
    static bool known(const int*) { return true; }
    static bool quietEligible(const int*) { return true; }
    static bool requiresResolution(const int*) { return false; }
  };

  /** Conservative transfer for FV cell means. */
  template<int N, int V, typename C, typename P>
  struct FVTransferPolicy {
    static constexpr bool IsFiniteVolume = true;
    static void restrict(const P* const* children, P* parent, const double*) {
      amr_detail::restrictVolumes<N,V>(children, parent);
    }
    static void prolong(const P* parent, P* child, const double*, int position) {
      amr_detail::interpolateVolumes<N,V>(parent, child, position);
    }
    static bool polynomialPasses(const P*, const P*) { return true; }
  };

  /** DG transfer with a continuous round-trip error bound. */
  template<int N, int V, typename C, typename P>
  struct DGTransferPolicy {
    static constexpr bool IsFiniteVolume = false;
    static void restrict(const P* const* children, P* parent, const double* matrix) {
      amr_detail::restrictDGCell<N,V>(children, parent, matrix);
    }
    static void prolong(const P* parent, P* child, const double* matrix, int position) {
      amr_detail::interpolateDGCell<N,V>(parent, child, matrix, position);
    }
    static bool polynomialPasses(const P* fine, const P* restored) {
      return amr_detail::transferPolynomialPasses<N,V,C>(fine, restored);
    }
  };

  template<int N, int V, typename C, typename P>
  using NativeTransferPolicy = std::conditional_t<C::IsFiniteVolume,
    FVTransferPolicy<N,V,C,P>, DGTransferPolicy<N,V,C,P>>;
} // namespace exahype2::amr
