// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "AssessmentPolicy.h"
#include "AMRMath.h"
#include "tarch/accelerator/ChunkedMemoryPool.h"

// The parts that a criterion is put together from: how to read the solution of a cell, how to move it
// between a parent and its children, and what a limiter adds. AMRCore takes them as template arguments.
TARGET_DECLARE
namespace exahype2::amr {
  /** How the values of a cell are stored: at the DG nodes or as averages over finite volumes. */
  enum class AcceptedRepresentation { DGNodal, FVCellAverages };

  /** The accepted solution of one cell, that is its solution after a completed time step: the values, the
   * number of nodes or volumes per axis and how the values are stored. */
  template<typename P>
  struct AcceptedCellView {
    const P* values{nullptr};
    int pointsPerAxis{0};
    AcceptedRepresentation representation{AcceptedRepresentation::DGNodal};
    [[nodiscard]] GPU_CALLABLE_METHOD int numberOfStates() const { return peano4::power(pointsPerAxis, DIMENSIONS); }
    [[nodiscard]] GPU_CALLABLE_METHOD bool isFiniteVolume() const { return representation == AcceptedRepresentation::FVCellAverages; }
  };

  /** Stands in for the limiter statuses of a solver without a limiter. No cell has a status. */
  struct NoStatusView {
    [[nodiscard]] GPU_CALLABLE_METHOD const int* getChunk(int) const { return nullptr; }
  };

  /** Reads the accepted solution of a solver without a limiter: the FV averages or DG nodal values of the cell. */
  template<int N, int V, typename C, typename P>
  struct NativeAcceptedSolutionAdapter {
    using Subcells = typename tarch::accelerator::ChunkedMemoryPool<P>::ConstView;
    static constexpr int SubcellsPerAxis = N;
    static constexpr bool HasHybridSubcells = false;
    static GPU_CALLABLE_METHOD AcceptedCellView<P> accepted(int, const P* state, const int*, Subcells) {
      return {state, N, C::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal};
    }
  };

  /** Reads the accepted solution of ADER-DG with the subcell limiter: the subcell averages where the limiter
   * took over, the DG nodal values elsewhere. */
  template<int N, int V, typename P, typename Layout>
  struct HybridAcceptedSolutionAdapter {
    using LimiterAdapter = Layout;
    using Subcells = typename tarch::accelerator::ChunkedMemoryPool<P>::ConstView;
    static constexpr int SubcellsPerAxis = Layout::SubcellsPerAxis;
    static constexpr bool HasHybridSubcells = true;
    static GPU_CALLABLE_METHOD AcceptedCellView<P> accepted(int slot, const P* state, const int* status, Subcells subcells) {
      const auto role = Layout::classify(status);
      if (not Layout::isKnown(role)) return {};
      if (not Layout::usesNativeFV(role)) return {state, N, AcceptedRepresentation::DGNodal};
      constexpr int values = peano4::power(SubcellsPerAxis, DIMENSIONS) * V;
      if (subcells.segments == nullptr or subcells.paddedChunkSize < values) return {};
      return {subcells.getChunk(slot), SubcellsPerAxis, AcceptedRepresentation::FVCellAverages};
    }
  };

  /** The answers about the limiter for a solver that has none: every cell may coarsen and none has to
   * refine. SubcellLimiterEvidence explains the questions. */
  struct NoLimiterEvidence {
    static GPU_CALLABLE_METHOD bool available(const int*) { return true; }
    static GPU_CALLABLE_METHOD bool known(const int*) { return true; }
    static GPU_CALLABLE_METHOD bool quietEligible(const int*) { return true; }
    static GPU_CALLABLE_METHOD bool requiresResolution(const int*) { return false; }
  };

  /** Moves FV averages between a parent and its children. Both directions keep the mean. */
  template<int N, int V, typename C, typename P>
  struct FVTransferPolicy {
    static constexpr bool IsFiniteVolume = true;
    static GPU_CALLABLE_METHOD void restrict(const P* const* children, P* parent, const double*) {
      amr_detail::restrictVolumes<N,V>(children, parent);
    }
    static GPU_CALLABLE_METHOD void prolong(const P* parent, P* child, const double*, int position) {
      amr_detail::interpolateVolumes<N,V>(parent, child, position);
    }
    static GPU_CALLABLE_METHOD bool polynomialPasses(const P*, const P*) { return true; }
  };

  /** Moves a DG polynomial between a parent and its children, and checks how far a child differs from the
   * polynomial that comes back from the parent. */
  template<int N, int V, typename C, typename P>
  struct DGTransferPolicy {
    static constexpr bool IsFiniteVolume = false;
    static GPU_CALLABLE_METHOD void restrict(const P* const* children, P* parent, const double* matrix) {
      amr_detail::restrictDGCell<N,V>(children, parent, matrix);
    }
    static GPU_CALLABLE_METHOD void prolong(const P* parent, P* child, const double* matrix, int position) {
      amr_detail::interpolateDGCell<N,V>(parent, child, matrix, position);
    }
    static GPU_CALLABLE_METHOD bool polynomialPasses(const P* fine, const P* restored) {
      return amr_detail::transferPolynomialPasses<N,V,C>(fine, restored);
    }
  };

  /** The transfer of a solver without a limiter: for FV or for DG, as Config says. */
  template<int N, int V, typename C, typename P>
  using NativeTransferPolicy = std::conditional_t<C::IsFiniteVolume,
    FVTransferPolicy<N,V,C,P>, DGTransferPolicy<N,V,C,P>>;
} // namespace exahype2::amr
TARGET_DECLARE_END
