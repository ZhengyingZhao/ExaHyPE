// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "AMRPolicies.h"
#include "NativeLimiterAdapter.h"
#include "exahype2/dg/limiting/Detector.h"
#include "exahype2/terms/PhysicalAdmissibility.h"

TARGET_DECLARE
namespace exahype2::amr {
  /**
   * Tells the criterion what the subcell limiter knows about a cell:
   *  - available: a status is stored for the cell.
   *  - known: the status is valid.
   *  - quietEligible: the cell may coarsen.
   *  - requiresResolution: the cell belongs to a limited region.
   *  - requestsRefinement: the cell computes on subcells and should refine.
   */
  template<typename Layout>
  struct SubcellLimiterEvidence {
    static_assert(LimiterAdapterProtocol<Layout>);
    static GPU_CALLABLE_METHOD bool available(const int* status) { return status != nullptr; }
    static GPU_CALLABLE_METHOD bool known(const int* status) { return Layout::isKnown(Layout::classify(status)); }
    static GPU_CALLABLE_METHOD bool quietEligible(const int* status) { return Layout::quietEligible(Layout::classify(status)); }
    static GPU_CALLABLE_METHOD bool requiresResolution(const int* status) { return Layout::requiresResolution(Layout::classify(status)); }
    static GPU_CALLABLE_METHOD bool requestsRefinement(const int* status) {
      const auto role = Layout::classify(status);
      return Layout::requiresResolution(role) and Layout::usesNativeFV(role);
    }
  };

  /** Is a DG cell physically admissible? Its values have to be finite. If the solver has an admissibility
   * check, the cell has to pass it at its DG nodes, as subcell averages and at the Lobatto points.
   */
  template <typename Solver, typename P>
  GPU_CALLABLE_METHOD bool parentPhysicallyAdmissible(
    const P* Q, int numberOfStates,
    const tarch::la::Vector<DIMENSIONS, double>& x, const tarch::la::Vector<DIMENSIONS, double>& h,
    const typename Solver::RuntimeParameters& parameters
  ) {
    constexpr int N = Solver::NodesPerAxis, V = Solver::NumberOfData;
    constexpr int Nodes = peano4::power(N, DIMENSIONS);
    if (Q == nullptr or numberOfStates != Nodes) return false;
    for (int entry = 0; entry < Nodes * V; entry++) {
      if (not std::isfinite(static_cast<double>(Q[entry]))) return false;
    }
    if constexpr (not exahype2::terms::HasPhysicalAdmissibility<Solver, P>) {
      return true;
    } else {
      using Scheme = typename Solver::LimiterType::SubcellScheme;
      using Projection = typename Solver::template SubcellProjection<P>;
      constexpr int Subcells = Scheme::SubcellsPerAxis;
      constexpr int Volumes = peano4::power(Subcells, DIMENSIONS);
      constexpr auto nodesToSubcells = Projection::getNodesToSubcells1D();
      constexpr auto nodesToLobatto = Projection::getNodesToLobatto1D();
      if (not detail::projectionIsSafe<N, Subcells, V>(Q, nodesToSubcells.data())
          or not detail::projectionIsSafe<N, N, V>(Q, nodesToLobatto.data())
          or not Solver::isPhysicallyAdmissible(Q, Nodes, x, h, parameters)) return false;
      P subcells[Volumes * V], lobatto[Nodes * V];
      exahype2::dg::projectTensorProduct<DIMENSIONS, N, Subcells, V,
        exahype2::dg::ProjectionRule::Butterfly>(Q, subcells, nodesToSubcells.data());
      exahype2::dg::projectTensorProduct<DIMENSIONS, N, N, V,
        exahype2::dg::ProjectionRule::Butterfly>(Q, lobatto, nodesToLobatto.data());
      for (int entry = 0; entry < Volumes * V; entry++) {
        if (not std::isfinite(static_cast<double>(subcells[entry]))) return false;
      }
      if (not Solver::isPhysicallyAdmissible(subcells, Volumes, x, h, parameters)) return false;
      for (int node = 0; node < Nodes; node++) {
        for (int entry = 0; entry < V; entry++) {
          if (not std::isfinite(static_cast<double>(lobatto[node * V + entry]))) return false;
        }
        if (not Solver::isPhysicallyAdmissible(lobatto + node * V, 1, x, h, parameters)) return false;
      }
      return true;
    }
  }

  /** Does a proposed parent respect the discrete maximum principle of the limiter? The quantities that the
   * limiter watches have to stay within their range over the children, widened by the limiter's tolerance. */
  template <typename Solver, typename P>
  GPU_CALLABLE_METHOD bool parentRespectsMaximumPrinciple(
    const P* parent, const P* const* children, double relaxation, double differencesScaling,
    const typename Solver::RuntimeParameters& parameters
  ) {
    using Detector = typename Solver::LimiterType::Detector;
    if constexpr (not Detector::UsesMaximumPrinciple) {
      return true;
    } else {
      using Projection = typename Solver::template SubcellProjection<P>;
      using Scheme = typename Solver::LimiterType::SubcellScheme;
      constexpr int N = Solver::NodesPerAxis, V = Solver::NumberOfData;
      constexpr int Subcells = Scheme::SubcellsPerAxis;
      constexpr auto nodesToSubcells = Projection::getNodesToSubcells1D();
      constexpr auto nodesToLobatto = Projection::getNodesToLobatto1D();
      const double parameterLimit = static_cast<double>(std::numeric_limits<P>::max());
      if (parent == nullptr or children == nullptr
          or not std::isfinite(relaxation) or relaxation < 0.0
          or not std::isfinite(differencesScaling) or differencesScaling < 0.0
          or relaxation > parameterLimit or differencesScaling > parameterLimit) return false;
      const P specifiedRelaxation = static_cast<P>(relaxation);
      const P specifiedScaling = static_cast<P>(differencesScaling);
      if (not detail::projectionIsSafe<N, Subcells, V>(parent, nodesToSubcells.data())
          or not detail::projectionIsSafe<N, N, V>(parent, nodesToLobatto.data())) return false;
      P reference[Detector::ExtremaPerCell], childExtrema[Detector::ExtremaPerCell];
      Detector::clearExtrema(reference);
      for (int child = 0; child < peano4::ChildrenPerCell; child++) {
        if (not detail::projectionIsSafe<N, Subcells, V>(children[child], nodesToSubcells.data())
            or not detail::projectionIsSafe<N, N, V>(children[child], nodesToLobatto.data())) return false;
        Detector::reduceExtrema(
          childExtrema, children[child], nodesToLobatto.data(), nodesToSubcells.data(), parameters
        );
        for (int entry = 0; entry < Detector::ExtremaPerCell; entry++) {
          if (not std::isfinite(static_cast<double>(childExtrema[entry]))) return false;
        }
        Detector::widenExtrema(reference, childExtrema);
      }
      int scratch[exahype2::dg::limiting::NumberOfStatusEntries]{};
      Detector::openInitialStatus(
        scratch, parent, reference,
        nodesToLobatto.data(), nodesToSubcells.data(),
        specifiedRelaxation, specifiedScaling, parameters
      );
      return scratch[static_cast<int>(exahype2::dg::limiting::StatusEntry::WasTroubled)] == 0;
    }
  }

} // namespace exahype2::amr
TARGET_DECLARE_END
