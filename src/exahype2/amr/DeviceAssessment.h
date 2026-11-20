// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "AMRCore.h"

namespace exahype2::amr::device_detail {
  /** Mesh records and refinement decisions used by the device kernels. */
  template <int N, int V, typename P>
  struct Storage {
    using Coordinates           = tarch::la::Vector<DIMENSIONS, double>;
    using Key                   = std::array<int, DIMENSIONS + 1>;
    static constexpr int Values = peano4::power(N, DIMENSIONS) * V;
    struct Cell {
      Key         key{};
      int         slot{-1}, family{-1}, firstFace{0}, faceCount{0};
      Coordinates centre{}, width{};
      const P*    state{nullptr};
      const int*  status{nullptr};
      double      quiet{0.}, acceptedQuiet{-1.}, crossing{0.};
      bool        moving{false}, complete{false};
    };
    struct Face {
      int    own{-1}, other{-1}, normal{-1}, side{-1};
      double value{0.};
    };
    /** Leaf cells that share a parent. Coarsening requires all siblings. */
    struct Family {
      std::array<int, peano4::ChildrenPerCell> slots{};
      bool                                     complete{false};
    };
    struct ResolutionEvidence {
      double value{0.};
      bool   complete{true};
    };
    /** The proposed parent solution and the checks it has passed on the current mesh. */
    struct FamilyPlan {
      std::array<P, Values> parent{};
      Key                   key{};
      bool                  sizeVetoed{false}, transferSafe{false}, neighboursSafe{false}, eligible{false};
    };
    /** How long a leaf has stayed resolved, and how long a wave takes to cross it. */
    struct QuietHistory {
      double quiet{-1.}, crossing{0.};
      bool   moving{false};
    };
    struct FamilyResult {
      bool sizeVetoed{false}, transferSafe{false}, neighboursSafe{false}, eligible{false}, passed{false};
    };
  };

  TARGET_DECLARE
  /** The refinement and coarsening tests for one solver's accepted solution. */
  template <int N, int V, typename C, typename P, typename A, typename T, typename E, typename S, typename Callbacks>
  struct Assessment {
    using Data               = Storage<N, V, P>;
    using Coordinates        = typename Data::Coordinates;
    using Key                = typename Data::Key;
    using Cell               = typename Data::Cell;
    using Face               = typename Data::Face;
    using Family             = typename Data::Family;
    using FamilyPlan         = typename Data::FamilyPlan;
    using FamilyResult       = typename Data::FamilyResult;
    using ResolutionEvidence = typename Data::ResolutionEvidence;
    using Geometry           = std::conditional_t<
                T::IsFiniteVolume,
                detail::retained_parent::NoGeometry,
                detail::retained_parent::Geometry<N, peano4::SubdivisionFactor>
              >;
    static constexpr int            Nodes = peano4::power(N, DIMENSIONS), Values = Nodes * V;
    static constexpr int            MaxPoints = N > A::SubcellsPerAxis ? N : A::SubcellsPerAxis;
    tarch::Span<Cell>               _cells;
    tarch::Span<Face>               _faces;
    tarch::Span<const Family>       _families;
    tarch::Span<ResolutionEvidence> _details;
    tarch::Span<FamilyPlan>         _plans;
    tarch::Span<FamilyResult>       _familyResults;
    typename tarch::accelerator::ChunkedMemoryPool<P>::ConstView _subcells;
    peano4::CellConnectivity                                     _mesh;
    const double*                                                _nodes{nullptr};
    const double*                                                _restriction{nullptr};
    const double*                                                _prolongation{nullptr};
    const Geometry*                                              _retainedParentGeometry{nullptr};
    double    _referenceTime{1.}, _restrictionBound{0.}, _prolongationBound{0.};
    double    _time{0.}, _dt{0.}, _maxPatchSize{0.};
    Callbacks _callbacks;

    GPU_CALLABLE_METHOD static constexpr bool predictionMode() {
      if constexpr (requires { C::UsePredictionCriterion; }) {
        return C::UsePredictionCriterion;
      }
      return false;
    }
    GPU_CALLABLE_METHOD static constexpr bool cacheAcceptedFaces() {
      if constexpr (requires { C::CacheAcceptedFaces; }) {
        return predictionMode() and C::CacheAcceptedFaces;
      }
      return predictionMode();
    }
    GPU_CALLABLE_METHOD static bool finite(const P* Q) {
      if (Q == nullptr) {
        return false;
      }
      for (int entry = 0; entry < Values; entry++) {
        if (not std::isfinite(static_cast<double>(Q[entry]))) {
          return false;
        }
      }
      return true;
    }
    GPU_CALLABLE_METHOD static bool finite(const AcceptedCellView<P>& view) {
      if (view.values == nullptr or view.pointsPerAxis <= 0) {
        return false;
      }
      for (int entry = 0; entry < view.numberOfStates() * V; entry++) {
        if (not std::isfinite(static_cast<double>(view.values[entry]))) {
          return false;
        }
      }
      return true;
    }
    /** Returns the values the solver kept for this cell, including FV subcells when the limiter used them. */
    GPU_CALLABLE_METHOD AcceptedCellView<P> accepted(const Cell& cell) const {
      return A::accepted(cell.slot, cell.state, cell.status, _subcells);
    }
    GPU_CALLABLE_METHOD static double localDetail(const P* Q) {
      return localDetail(
        {Q, N, T::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal}
      );
    }
    GPU_CALLABLE_METHOD static double localDetail(const AcceptedCellView<P>& view) {
      if (view.isFiniteVolume()) {
        return resolution_detail::fvResolutionDetail<V, C>(
          view.values,
          view.pointsPerAxis,
          amr_detail::resolutionTolerance<C>()
        );
      }
      if (not finite(view)) {
        return std::numeric_limits<double>::infinity();
      }
      if (view.pointsPerAxis != N) {
        return std::numeric_limits<double>::infinity();
      }
      return amr_detail::highestOrderDetailImpl<S::NativeArithmetic, true, N, V, C>(
        view.values,
        amr_detail::resolutionTolerance<C>()
      );
    }
    /** An observable over its error budget raises the score above the refinement threshold. */
    GPU_CALLABLE_METHOD static double withObservation(double detail, double ratio) {
      double scaled = 0.0;
      if (not std::isfinite(ratio) or ratio < 0.0
          or not native_neighbour_detail::product(ratio, amr_detail::resolutionTolerance<C>(), scaled)) {
        return std::numeric_limits<double>::infinity();
      }
      if (ratio > 1.0) {
        scaled = std::max(
          scaled,
          std::nextafter(amr_detail::resolutionTolerance<C>(), std::numeric_limits<double>::infinity())
        );
      }
      return std::max(detail, scaled);
    }
    GPU_CALLABLE_METHOD double acceptedLocalDetail(const AcceptedCellView<P>& view) const {
      double result =
        predictionMode() ? prediction_detail::deviceLocalDetail<N, V, C, MaxPoints>(view, _nodes) : localDetail(view);
      if constexpr (observation_detail::count<C>() > 0) {
        result = withObservation(result, observation_detail::deviceLocalRatio<N, V, C, MaxPoints>(view, _nodes, false));
      }
      return result;
    }
    /** Rates how closely the solutions on either side of a face agree, including at a change in grid level. */
    GPU_CALLABLE_METHOD double faceDetail(
      const AcceptedCellView<P>& own,
      const AcceptedCellView<P>& other,
      const Key&                 ownKey,
      const Key&                 otherKey,
      int                        normal,
      int                        side
    ) const {
      if constexpr (predictionMode()) {
        double result =
          prediction_detail::deviceFaceDetail<N, V, C, MaxPoints>(own, other, ownKey, otherKey, _nodes, normal, side);
        if constexpr (observation_detail::count<C>() > 0) {
          result = withObservation(
            result,
            observation_detail::deviceFaceRatio<N, V, C, MaxPoints>(own, other, ownKey, otherKey, _nodes, normal, side)
          );
        }
        return result;
      } else {
        return resolution_detail::deviceFaceDetail<N, V, C, MaxPoints>(
          own,
          other,
          ownKey,
          otherKey,
          _nodes,
          normal,
          side
        );
      }
    }
    GPU_CALLABLE_METHOD ResolutionEvidence cellDetail(const Cell& cell) const { return _details[cell.slot]; }
    /** Rates how well the cell is resolved and records whether all required face neighbours were checked. */
    GPU_CALLABLE_METHOD ResolutionEvidence computeCellDetail(const Cell& cell) const {
      ResolutionEvidence result{acceptedLocalDetail(accepted(cell)), true};
      if constexpr (S::CheckNeighbours) {
        if (result.value > amr_detail::resolutionTolerance<C>()) {
          result.complete = false;
          return result;
        }
        for (int i = 0; i < cell.faceCount; i++) {
          if (result.value > amr_detail::resolutionTolerance<C>()) {
            result.complete = false;
            break;
          }
          const auto& face = _faces[cell.firstFace + i];
          if (face.other < 0 or face.other >= _cells.size) {
            result.complete = false;
            continue;
          }
          const auto view = accepted(_cells[face.other]);
          if (view.values == nullptr or view.pointsPerAxis <= 0) {
            result.complete = false;
            continue;
          }
          result.value = std::max(result.value, face.value);
        }
        result.complete = result.complete and cell.complete;
      }
      return result;
    }
    /** Returns the cell's accumulated quiet time in wave crossing times, or -1 if this step is ineligible. */
    GPU_CALLABLE_METHOD double computeQuietProgress(Cell& cell, double time, double dt) const {
      cell.crossing = std::numeric_limits<double>::infinity();
      cell.moving   = false;
      if (not E::quietEligible(cell.status)) {
        return -1.0;
      }
      const auto view = accepted(cell);
      if constexpr (S::CheckQuietDetail and not predictionMode()) {
        const auto detail = cellDetail(cell);
        if (not detail.complete or detail.value > amr_detail::resolutionTolerance<C>()
            or (not _callbacks.admissible(view.values, view.numberOfStates(), cell.centre, cell.width))) {
          return -1.0;
        }
      } else if (not finite(view)
                 or (not _callbacks.admissible(view.values, view.numberOfStates(), cell.centre, cell.width))) {
        return -1.0;
      }
      double crossing = std::numeric_limits<double>::infinity();
      bool   moving   = false;
      for (int node = 0; node < view.numberOfStates(); node++) {
        auto index     = tarch::la::Vector<DIMENSIONS, int>(0);
        int  remaining = node;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          index(axis) = remaining % view.pointsPerAxis;
          remaining /= view.pointsPerAxis;
        }
        Coordinates point;
        if constexpr (not S::NativeArithmetic and not T::IsFiniteVolume) {
          point = exahype2::dg::getQuadraturePoint(cell.centre, cell.width, index, _nodes);
        } else {
          for (int axis = 0; axis < DIMENSIONS; axis++) {
            const double position = view.isFiniteVolume() ? (index(axis) + 0.5) / view.pointsPerAxis : _nodes[index(axis)];
            point(axis) = cell.centre(axis) + (position - 0.5) * cell.width(axis);
          }
        }
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          const double speed = _callbacks.waveSpeed(view.values + node * V, point, cell.width, time, dt, axis);
          if (not std::isfinite(speed) or speed < 0.0) {
            return -1.0;
          }
          if (speed == 0.0) {
            continue;
          }
          moving             = true;
          const double width = cell.width(axis);
          const double limit = std::numeric_limits<double>::max();
          const double tau =
            speed < 1.0 and width > limit * speed ? std::numeric_limits<double>::infinity() : width / speed;
          crossing = std::min(crossing, tau);
        }
      }
      cell.crossing = crossing;
      cell.moving   = moving;
      return amr_detail::advanceQuiet(cell.quiet, crossing, moving, dt, _referenceTime);
    }
    /** Tests whether one parent cell can represent its children within the allowed error.
     * On success, parent holds the coarsened solution and parentKey identifies its grid cell. */
    GPU_CALLABLE_METHOD bool transferPasses(const Family& family, P* parent, Key& parentKey) const {
      if (not family.complete or _restrictionBound <= 0.0 or _prolongationBound <= 0.0) {
        return false;
      }
      if (not amr_detail::transfer_bound::arithmeticSupported()) {
        return false;
      }
      const P*                          children[peano4::ChildrenPerCell];
      observation_detail::Stencil<V, C> observedTransfer;
      const double                      limit = static_cast<double>(std::numeric_limits<P>::max());
      for (int child = 0; child < peano4::ChildrenPerCell; child++) {
        const auto& cell = _cells[family.slots[child]];
        if (not E::quietEligible(cell.status)) {
          return false;
        }
        if constexpr (S::CheckQuietDetail or predictionMode()) {
          const auto detail = cellDetail(cell);
          if (not detail.complete or detail.value > amr_detail::resolutionTolerance<C>()) {
            return false;
          }
        }
        if constexpr (not predictionMode()) {
          if (cell.acceptedQuiet < C::QuietTime) {
            return false;
          }
        }
        const auto view = accepted(cell);
        if constexpr (predictionMode()) {
          if (not finite(view)
              or not _callbacks.admissible(view.values, view.numberOfStates(), cell.centre, cell.width)) {
            return false;
          }
        }
        // The transfer operator expects the layout of the solver's main solution.
        if (view.values != cell.state or view.pointsPerAxis != N or view.isFiniteVolume() != T::IsFiniteVolume) {
          return false;
        }
        children[child] = view.values;
        if constexpr (observation_detail::count<C>() > 0) {
          if (not observedTransfer.add(view)) {
            return false;
          }
        }
        for (int entry = 0; entry < Values; entry++) {
          if (not std::isfinite(static_cast<double>(children[child][entry]))
              or std::abs(static_cast<double>(children[child][entry])) > limit / _restrictionBound) {
            return false;
          }
        }
      }
      P restored[Values];
      T::restrict(children, parent, _restriction);
      if (not finite(parent)) {
        return false;
      }
      for (int entry = 0; entry < Values; entry++) {
        if (std::abs(static_cast<double>(parent[entry])) > limit / _prolongationBound) {
          return false;
        }
      }
      const auto& first = _cells[family.slots[0]];
      Coordinates width, centre;
      parentKey         = {};
      parentKey[0]      = first.key[0] - 1;
      const auto offset = _mesh.getDomainOffset();
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        parentKey[axis + 1] = first.key[axis + 1] / peano4::SubdivisionFactor;
        width(axis)         = _mesh.getWidthOfLevel(first.key[0] - 1, axis);
        centre(axis)        = offset(axis) + (static_cast<double>(parentKey[axis + 1]) + 0.5) * width(axis);
        if (not std::isfinite(width(axis)) or width(axis) <= 0.0 or not std::isfinite(centre(axis))) {
          return false;
        }
      }
      if (not _callbacks.admissible(parent, Nodes, centre, width)) {
        return false;
      }
      if (not _callbacks.transferAdmissible(parent, children)) {
        return false;
      }
      if constexpr (not predictionMode()) {
        if (localDetail(parent) > amr_detail::resolutionTolerance<C>()) {
          return false;
        }
      }
      double           rms     = 0.0;
      constexpr double samples = static_cast<double>(peano4::ChildrenPerCell) * Nodes;
      // Bytes per thread for caching interpolation along the first axis.
      constexpr std::size_t cacheBytes = 0;
      using Prepared                   = std::conditional_t<
                          T::IsFiniteVolume,
                          detail::retained_parent::NoGeometry,
                          detail::retained_parent::PreparedParent<N, V, DIMENSIONS, peano4::SubdivisionFactor, P, cacheBytes>
                        >;
      using ObservedPrepared = std::conditional_t<
        not T::IsFiniteVolume and (observation_detail::count<C>() > 0),
        Prepared,
        detail::retained_parent::NoGeometry
      >;
      [[maybe_unused]] Prepared         preparedParent;
      [[maybe_unused]] ObservedPrepared observedParent;
      [[maybe_unused]] bool             parentPrepared = false, observationsPrepared = false;
      for (int child = 0; child < peano4::ChildrenPerCell; child++) {
        T::prolong(parent, restored, _prolongation, child);
        if (not finite(restored)) {
          return false;
        }
        for (int node = 0; node < Nodes; node++) {
          std::array<double, V> residuals;
          for (int entry = 0; entry < V; entry++) {
            const double scale = C::stateScale(entry);
            const double a     = static_cast<double>(children[child][node * V + entry]);
            const double b     = static_cast<double>(restored[node * V + entry]);
            if (not std::isfinite(scale) or scale <= 0.0) {
              return false;
            }
            residuals[entry] =
              amr_detail::transfer_bound::magnitude(amr_detail::transfer_bound::difference(a, b, scale));
            if (not std::isfinite(residuals[entry]) or residuals[entry] > C::TransferTolerance) {
              return false;
            }
            if constexpr (predictionMode()) {
              rms = amr_detail::guardedHypot(rms, residuals[entry] / std::sqrt(samples));
            }
          }
          if (not amr_detail::transfer_bound::normPasses<V>(residuals, C::TransferTolerance)) {
            return false;
          }
        }
        if (not T::polynomialPasses(children[child], restored)) {
          return false;
        }
        if constexpr (observation_detail::count<C>() > 0) {
          if (not observedTransfer.template transferPasses<N>(children[child], restored, T::IsFiniteVolume)) {
            return false;
          }
        }
        if constexpr (not T::IsFiniteVolume) {
          // Coarsening keeps the parent polynomial. Its error matters between the child nodes too.
          std::array<double, V> scales;
          for (int component = 0; component < V; component++) {
            scales[component] = C::stateScale(component);
          }
          amr_detail::transfer_bound::Field<N, V, DIMENSIONS> residual;
          if (not parentPrepared) {
            if (not preparedParent.prepare(parent, *_retainedParentGeometry, scales)) {
              return false;
            }
            parentPrepared = true;
          }
          if (not preparedParent.residual(children[child], child, residual)
              or not amr_detail::transferPolynomialIntervalsPasses<N, V, C>(residual)) {
            return false;
          }
          if constexpr (observation_detail::count<C>() > 0) {
            // Observable weights apply to the original, unscaled fields.
            bool unitScales = true;
            for (double scale : scales) {
              unitScales = unitScales and scale == 1.;
            }
            if (unitScales) {
              if (not observedTransfer.template transferIntervalsPasses<N>(residual, false)) {
                return false;
              }
            } else {
              scales.fill(1.);
              if (not observationsPrepared) {
                if (not observedParent.prepare(parent, *_retainedParentGeometry, scales)) {
                  return false;
                }
                observationsPrepared = true;
              }
              if (not observedParent.residual(children[child], child, residual)
                  or not observedTransfer.template transferIntervalsPasses<N>(residual, false)) {
                return false;
              }
            }
          }
        }
      }
      if constexpr (predictionMode()) {
        if (not std::isfinite(rms) or rms > amr_detail::resolutionTolerance<C>()) {
          return false;
        }
      }
      return true;
    }
    /** Can a limiter-marked cell stay at its current size? Its monitored fields must be constant
     * across the cell and its face neighbours, up to projection roundoff. */
    GPU_CALLABLE_METHOD bool locallyConstant(const Cell& cell) const {
      if constexpr (not S::UseLocalConstancy) {
        return false;
      } else {
        if (not amr_detail::transfer_bound::arithmeticSupported() or _subcells.segments == nullptr) {
          return false;
        }
        constexpr int SubcellsPerAxis = A::SubcellsPerAxis;
        static_assert(SubcellsPerAxis > 0);
        constexpr int Subcells = peano4::power(SubcellsPerAxis, DIMENSIONS);
        if (_subcells.paddedChunkSize < Subcells * V) {
          return false;
        }
        constexpr double operations = 2.0 * SubcellsPerAxis * DIMENSIONS;
        constexpr double product    = operations * (std::numeric_limits<P>::epsilon() / 2.0);
        if constexpr (product >= 1.0) {
          return false;
        }
        constexpr double      gamma  = product / (1.0 - product);
        const double          budget = std::min(C::TransferTolerance, amr_detail::resolutionTolerance<C>());
        std::array<double, V> lower, upper, magnitudes{};
        lower.fill(std::numeric_limits<double>::infinity());
        upper.fill(-std::numeric_limits<double>::infinity());
        const auto needsConstancy = [](int entry) {
          if (amr_detail::resolutionComponent<C>(entry)) {
            return true;
          }
          if constexpr (observation_detail::count<C>() > 0) {
            for (int observation = 0; observation < observation_detail::count<C>(); observation++) {
              if (C::observableWeight(observation, entry) != 0.0) {
                return true;
              }
            }
          }
          return false;
        };
        const auto matches = [&](const P* values, int states) {
          if (values == nullptr or states <= 0) {
            return false;
          }
          for (int node = 0; node < states; node++) {
            for (int entry = 0; entry < V; entry++) {
              const double value = values[node * V + entry];
              const double limit = std::numeric_limits<double>::max();
              if (not std::isfinite(value)) {
                return false;
              }
              if (not needsConstancy(entry)) {
                continue;
              }
              lower[entry] = std::min(lower[entry], value);
              upper[entry] = std::max(upper[entry], value);
              if (upper[entry] > 0.0 and lower[entry] < 0.0 and upper[entry] > limit + lower[entry]) {
                return false;
              }
              const double difference = upper[entry] - lower[entry];
              const double magnitude  = std::min(std::abs(lower[entry]), std::abs(upper[entry]));
              const double scale      = C::stateScale(entry);
              if (difference > gamma * magnitude or not std::isfinite(scale) or scale <= 0.0
                  or (scale < 1.0 and difference > limit * scale)) {
                return false;
              }
              magnitudes[entry] = amr_detail::transfer_bound::magnitude(
                amr_detail::transfer_bound::difference(upper[entry], lower[entry], scale)
              );
              if (not std::isfinite(magnitudes[entry]) or magnitudes[entry] > budget) {
                return false;
              }
            }
          }
          return true;
        };
        const auto inspect = [&](const Cell& observed) {
          if (not E::known(observed.status)) {
            return false;
          }
          const auto view = accepted(observed);
          if (view.values == nullptr or view.pointsPerAxis <= 0) {
            return false;
          }
          if (not matches(observed.state, Nodes)) {
            return false;
          }
          double detail;
          if constexpr (predictionMode() and requires { C::resolutionComponent(0); }) {
            std::array<P, Values> selected{};
            for (int node = 0; node < Nodes; node++) {
              for (int component = 0; component < V; component++) {
                if (needsConstancy(component)) {
                  selected[node * V + component] = observed.state[node * V + component];
                }
              }
            }
            detail = localDetail(selected.data());
          } else {
            detail = localDetail(observed.state);
          }
          if (detail > amr_detail::resolutionTolerance<C>()) {
            return false;
          }
          if (view.isFiniteVolume() and not matches(view.values, view.numberOfStates())) {
            return false;
          }
          return _callbacks.admissible(observed.state, Nodes, observed.centre, observed.width)
            and (not view.isFiniteVolume()
                 or _callbacks.admissible(view.values, view.numberOfStates(), observed.centre, observed.width));
        };
        if (not inspect(cell)) {
          return false;
        }
        bool       matchesNeighbours = true;
        const bool complete          = cell.complete;
        for (int i = 0; i < cell.faceCount and matchesNeighbours; i++) {
          const int slot = _faces[cell.firstFace + i].other;
          if (slot < 0 or slot >= _cells.size or _cells[slot].slot != slot) {
            matchesNeighbours = false;
          } else {
            matchesNeighbours = inspect(_cells[slot]);
          }
        }
        return complete and matchesNeighbours and amr_detail::transfer_bound::normPasses<V>(magnitudes, budget);
      }
    }

    /** Checks the proposed parent's faces against the current neighbours.
     * With onlyProposed set, the comparison uses neighbouring parents that are also eligible to coarsen. */
    GPU_CALLABLE_METHOD bool parentPassesNeighbours(
      const Family& family,
      const P*      parent,
      const Key&    key,
      bool          onlyProposed
    ) const {
      if constexpr (not S::CheckNeighbours) {
        return true;
      }
      const AcceptedCellView<P> proposed{
        parent,
        N,
        T::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal
      };
      for (int childSlot : family.slots) {
        const auto& child = _cells[childSlot];
        if (not child.complete) {
          return false;
        }
        for (int i = 0; i < child.faceCount; i++) {
          const auto& face    = _faces[child.firstFace + i];
          bool        sibling = false;
          for (int slot : family.slots) {
            sibling = sibling or slot == face.other;
          }
          if (sibling) {
            const int  position = child.key[face.normal + 1] % peano4::SubdivisionFactor;
            const bool exterior = face.side == 0 ? position == 0 : position == peano4::SubdivisionFactor - 1;
            if (not onlyProposed and exterior) {
              const double value = faceDetail(proposed, proposed, key, key, face.normal, face.side);
              if (not std::isfinite(value) or value > amr_detail::resolutionTolerance<C>()) {
                return false;
              }
            }
            continue;
          }
          if (face.other < 0 or face.other >= _cells.size) {
            return false;
          }
          const auto&         neighbour = _cells[face.other];
          AcceptedCellView<P> view;
          Key                 otherKey;
          if (not onlyProposed) {
            view     = accepted(neighbour);
            otherKey = neighbour.key;
            if (not finite(view)) {
              return false;
            }
          } else {
            if (neighbour.family < 0 or not _families[neighbour.family].complete) {
              continue;
            }
            const auto& trial = _plans[neighbour.family];
            if (not trial.eligible) {
              continue;
            }
            view     = {trial.parent.data(), N, proposed.representation};
            otherKey = trial.key;
          }
          const double value = faceDetail(proposed, view, key, otherKey, face.normal, face.side);
          if (not std::isfinite(value) or value > amr_detail::resolutionTolerance<C>()) {
            return false;
          }
        }
      }
      return true;
    }
    /** Records whether these siblings can coarsen while their neighbours are still at their current sizes. */
    GPU_CALLABLE_METHOD void prepareFamily(int family) const {
      auto& plan           = _plans[family];
      plan                 = FamilyPlan{};
      const auto& siblings = _families[family];
      if (siblings.complete) {
        const double width = tarch::la::max(_cells[siblings.slots[0]].width);
        if (std::isfinite(width) and width >= 0.
            and width <= std::numeric_limits<double>::max() / peano4::SubdivisionFactor
            and not(peano4::SubdivisionFactor * width <= _maxPatchSize)) {
          plan.sizeVetoed = true;
          return;
        }
      }
      plan.transferSafe   = transferPasses(siblings, plan.parent.data(), plan.key);
      bool parentResolved = true;
      if constexpr (predictionMode()) {
        if (plan.transferSafe) {
          const AcceptedCellView<P> proposed{
            plan.parent.data(),
            N,
            T::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal
          };
          const double value = acceptedLocalDetail(proposed);
          parentResolved     = std::isfinite(value) and value <= amr_detail::resolutionTolerance<C>();
        }
      }
      plan.neighboursSafe =
        plan.transferSafe and parentResolved and parentPassesNeighbours(siblings, plan.parent.data(), plan.key, false);
      plan.eligible = plan.transferSafe and plan.neighboursSafe;
      if (plan.eligible) {
        plan.eligible = peano4::SubdivisionFactor * tarch::la::max(_cells[siblings.slots[0]].width) <= _maxPatchSize;
      }
    }
    /** Chooses Refine, Keep or Erase for a leaf. Erase requires the whole sibling group to qualify. */
    GPU_CALLABLE_METHOD exahype2::RefinementCommand command(const Cell& cell) const {
      using Command = exahype2::RefinementCommand;
      if (cell.state == nullptr or not E::available(cell.status)) {
        return Command::Keep;
      }
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (not std::isfinite(cell.centre(axis)) or not std::isfinite(cell.width(axis)) or cell.width(axis) <= 0.) {
          return Command::Keep;
        }
      }
      if (E::requiresResolution(cell.status)) {
        if constexpr (predictionMode() and requires { E::requestsRefinement(cell.status); }) {
          if (E::requestsRefinement(cell.status)) {
            return locallyConstant(cell) ? Command::Keep : Command::Refine;
          }
        } else {
          return locallyConstant(cell) ? Command::Keep : Command::Refine;
        }
      }
      if constexpr (not predictionMode()) {
        if (not E::quietEligible(cell.status)) {
          return Command::Keep;
        }
      } else if constexpr (requires { E::known(cell.status); }) {
        if (not E::known(cell.status)) {
          return Command::Keep;
        }
      }
      const auto view = accepted(cell);
      if constexpr (predictionMode()) {
        if (view.values == nullptr or view.pointsPerAxis <= 0) {
          return Command::Keep;
        }
      }
      const auto detail = cellDetail(cell);
      if (detail.value > amr_detail::resolutionTolerance<C>()) {
        return Command::Refine;
      }
      if constexpr (S::CheckCommandPhysical) {
        if (not _callbacks.admissible(view.values, view.numberOfStates(), cell.centre, cell.width)) {
          return Command::Refine;
        }
      }
      if (not E::quietEligible(cell.status) or not detail.complete) {
        return Command::Keep;
      }
      return cell.family >= 0 and _familyResults[cell.family].passed ? Command::Erase : Command::Keep;
    }
    /** Chooses whether to refine a leaf to represent the initial solution. */
    GPU_CALLABLE_METHOD exahype2::RefinementCommand initialCommand(
      const P*           Q,
      const Coordinates& x,
      const Coordinates& h
    ) const {
      using Command = exahype2::RefinementCommand;
      if (Q == nullptr) {
        return Command::Keep;
      }
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (not std::isfinite(x(axis)) or not std::isfinite(h(axis)) or h(axis) <= 0.) {
          return Command::Keep;
        }
      }
      if (not finite(Q) or not _callbacks.admissible(Q, Nodes, x, h)) {
        return Command::Refine;
      }
      if constexpr (Callbacks::HasInitialDataCheck) {
        return _callbacks.initialDataPasses(Q, x, h) ? Command::Keep : Command::Refine;
      } else {
        bool unresolved = false;
        if constexpr (observation_detail::count<C>() > 0) {
          const AcceptedCellView<P> view{
            Q,
            N,
            T::IsFiniteVolume ? AcceptedRepresentation::FVCellAverages : AcceptedRepresentation::DGNodal
          };
          unresolved = observation_detail::deviceLocalRatio<N, V, C, MaxPoints>(view, nullptr, true) > 1.;
        }
        double value;
        if constexpr (predictionMode() and requires { C::resolutionComponent(0); }) {
          std::array<P, Values> selected{};
          for (int node = 0; node < Nodes; node++) {
            for (int component = 0; component < V; component++) {
              if (amr_detail::resolutionComponent<C>(component)) {
                selected[node * V + component] = Q[node * V + component];
              }
            }
          }
          value = localDetail(selected.data());
        } else {
          value = localDetail(Q);
        }
        return unresolved or value > amr_detail::resolutionTolerance<C>() ? Command::Refine : Command::Keep;
      }
    }
  };
  TARGET_DECLARE_END
} // namespace exahype2::amr::device_detail
