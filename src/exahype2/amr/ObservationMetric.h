// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "PredictionSensor.h"

// Linear observables: weighted sums of the values of a cell with tolerances of their own (LinearObservable
// in amr.py). This file turns the values of a cell into its observables, divides each by its tolerance, and
// runs the scores and the transfer check on the result. A result above one means that an observable exceeds
// its tolerance.
namespace exahype2::amr::observation_detail {
  /** Returns the number of observables of Config, zero if it has none. */
  template <typename C>
  GPU_CALLABLE_METHOD constexpr int count() {
    if constexpr (requires { C::NumberOfObservables; }) return C::NumberOfObservables;
    return 0;
  }

  /** Can the observables of Config be used? Each needs finite weights, at least one of them not zero, a
   * positive absolute tolerance and a relative tolerance that is not negative. */
  template <int V, typename C>
  GPU_CALLABLE_METHOD bool valid() {
    if constexpr (count<C>() == 0) return true;
    else if constexpr (count<C>() < 0 or not requires {
      C::observableWeight(0, 0);
      C::observableAbsoluteTolerance(0);
      C::observableRelativeTolerance(0);
    }) return false;
    else {
      for (int observation = 0; observation < count<C>(); observation++) {
        const double absolute = C::observableAbsoluteTolerance(observation);
        const double relative = C::observableRelativeTolerance(observation);
        if (not std::isfinite(absolute) or absolute <= 0.0
            or not std::isfinite(relative) or relative < 0.0) return false;
        bool nonzero = false;
        for (int component = 0; component < V; component++) {
          const double weight = C::observableWeight(observation, component);
          if (not std::isfinite(weight)) return false;
          nonzero = nonzero or weight != 0.0;
        }
        if (not nonzero) return false;
      }
      return true;
    }
  }

  /** The Config for the scores of observables. Observables are divided by their tolerances first, so every
   * scale and tolerance here is one. */
  template <typename C>
  struct UnitConfig {
    static GPU_CALLABLE_METHOD double stateScale(int) { return 1.0; }
    static constexpr double ResolutionTolerance = 1.0;
    static constexpr double TransferTolerance = 1.0;
    static GPU_CALLABLE_METHOD double highestModeWeight(int node) requires requires { C::highestModeWeight(node); } {
      return C::highestModeWeight(node);
    }
  };

  /** Collects the range of every observable over a group of cells: a cell and its neighbour, or the children
   * of one parent. The tolerance of an observable, its budget, is the absolute tolerance plus the relative
   * tolerance times this range.
   */
  template <int V, typename C>
  class Stencil {
    static constexpr int Observations = count<C>();
    static_assert(V > 0 and Observations >= 0);
    using Interval = detail::transfer_bound::Interval;
    std::array<double, V> _lower{}, _upper{};
    // The range is bounded from below, so that rounding cannot enlarge a tolerance.
    std::array<double, Observations> _minimumUpper{}, _maximumLower{};
    bool _valid = observation_detail::valid<V, C>();
    bool _empty = true;

    /** Returns bounds of one observable of one state. */
    template <typename P>
    GPU_CALLABLE_METHOD Interval observationInterval(const P* state, int observation) const {
      namespace bound = detail::transfer_bound;
      std::array<Interval, V> terms;
      for (int component = 0; component < V; component++) {
        const double weight = C::observableWeight(observation, component);
        const double value = static_cast<double>(state[component]);
        terms[component] = bound::mul({weight, weight}, {value, value});
      }
      return bound::canonicalSum<V>(terms);
    }

    /** Fills in the tolerance of every observable and returns the number of nodes or volumes of the cell, or
     * zero if something is not valid.
     */
    template <typename P>
    GPU_CALLABLE_METHOD int prepareProjection(AcceptedCellView<P> view, std::array<double, Observations>& budgets) const {
      if (not _valid or _empty or view.values == nullptr or view.pointsPerAxis <= 0) return 0;
      int states = 1;
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (states > std::numeric_limits<int>::max() / view.pointsPerAxis) return 0;
        states *= view.pointsPerAxis;
      }
      if (states > std::numeric_limits<int>::max() / std::max(V, Observations)) return 0;
      for (int observation = 0; observation < Observations; observation++) {
        budgets[observation] = budget(observation);
        if (not std::isfinite(budgets[observation]) or budgets[observation] <= 0.0) return 0;
      }
      return states;
    }

    /** Writes every observable of every node, measured from the middle of the collected range and divided by
     * its tolerance. */
    template <typename P>
    GPU_CALLABLE_METHOD bool projectValues(
      const P* values, int states, const std::array<double, Observations>& budgets, double* output
    ) const {
      for (int node = 0; node < states; node++) {
        std::array<double, V> differences;
        for (int component = 0; component < V; component++) {
          const double reference = 0.5 * _lower[component] + 0.5 * _upper[component];
          differences[component] = native_neighbour_detail::scaledDifference(
            static_cast<double>(values[node * V + component]), reference, 1.0);
          if (not std::isfinite(differences[component])) return false;
        }
        for (int observation = 0; observation < Observations; observation++) {
          std::array<double, V> terms;
          for (int component = 0; component < V; component++) {
            if (not native_neighbour_detail::product(C::observableWeight(observation, component),
                  differences[component], terms[component])) return false;
          }
          double value = 0.0;
          if (not resolution_detail::sumTerms<V>(terms, value)
              or not native_neighbour_detail::ratio(value, budgets[observation], value)) return false;
          output[node * Observations + observation] = value;
        }
      }
      return true;
    }

  public:
    GPU_CALLABLE_METHOD Stencil() {
      _lower.fill(INFINITY);
      _upper.fill(-INFINITY);
      _minimumUpper.fill(INFINITY);
      _maximumLower.fill(-INFINITY);
    }

    /** Adds the values of a cell to the ranges. */
    template <typename P>
    GPU_CALLABLE_METHOD bool add(AcceptedCellView<P> view) {
      if constexpr (Observations == 0) return true;
      else {
        if (not _valid or view.values == nullptr or view.pointsPerAxis <= 0
            or not detail::transfer_bound::arithmeticSupported()) return _valid = false;
        int states = 1;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          if (states > std::numeric_limits<int>::max() / view.pointsPerAxis) return _valid = false;
          states *= view.pointsPerAxis;
        }
        if (states > std::numeric_limits<int>::max() / V) return _valid = false;
        for (int node = 0; node < states; node++) {
          const auto* state = view.values + node * V;
          for (int component = 0; component < V; component++) {
            const double value = static_cast<double>(state[component]);
            if (not std::isfinite(value)) return _valid = false;
            _lower[component] = std::min(_lower[component], value);
            _upper[component] = std::max(_upper[component], value);
          }
          for (int observation = 0; observation < Observations; observation++) {
            const auto value = observationInterval(state, observation);
            if (not detail::transfer_bound::valid(value)) return _valid = false;
            _minimumUpper[observation] = std::min(_minimumUpper[observation], value.hi);
            _maximumLower[observation] = std::max(_maximumLower[observation], value.lo);
          }
        }
        _empty = false;
        return true;
      }
    }

    /** Returns the tolerance of an observable for the collected cells, or zero if it cannot be bounded. */
    GPU_CALLABLE_METHOD double budget(int observation) const {
      if constexpr (Observations == 0) return INFINITY;
      else {
        namespace bound = detail::transfer_bound;
        if (not _valid or _empty or observation < 0 or observation >= Observations) return 0.0;
        const double absolute = C::observableAbsoluteTolerance(observation);
        const double relative = C::observableRelativeTolerance(observation);
        if (relative == 0.0) return absolute;
        const auto range = bound::difference(_maximumLower[observation], _minimumUpper[observation], 1.0);
        if (not bound::valid(range)) return 0.0;
        const double variation = std::max(0.0, range.lo);
        if (variation == 0.0) return absolute;
        const auto allowance = bound::add({absolute, absolute},
          bound::mul({relative, relative}, {variation, variation}));
        return bound::valid(allowance) and allowance.lo > 0.0 ? allowance.lo : 0.0;
      }
    }

    /** Writes the observables of a cell, divided by their tolerances, to output. capacity is the size of output. */
    template <typename P>
    GPU_CALLABLE_METHOD bool project(AcceptedCellView<P> view, double* output, int capacity) const {
      if constexpr (Observations == 0) return true;
      else {
        std::array<double, Observations> budgets;
        const int states = prepareProjection(view, budgets);
        if (states == 0 or output == nullptr or capacity < states * Observations) return false;
        return projectValues(view.values, states, budgets, output);
      }
    }

    /** The same into a vector. */
    template <typename P>
    bool project(AcceptedCellView<P> view, std::vector<double>& output) const {
      if constexpr (Observations == 0) { output.clear(); return true; }
      else {
        std::array<double, Observations> budgets;
        const int states = prepareProjection(view, budgets);
        if (states == 0) return false;
        output.resize(static_cast<std::size_t>(states) * Observations);
        return projectValues(view.values, states, budgets, output.data());
      }
    }

    /** Do the observables stay within their tolerances when a parent replaces a child? residual bounds the
     * difference between child and parent at every node, in the units of the values. For DG the check holds
     * everywhere in the cell, not only at the nodes.
     */
    template <int N>
    GPU_CALLABLE_METHOD bool transferIntervalsPasses(const detail::transfer_bound::Field<N, V, DIMENSIONS>& residual, bool finiteVolume) const {
      if constexpr (Observations == 0) return true;
      else {
        namespace bound = detail::transfer_bound;
        if (not _valid or _empty or not bound::arithmeticSupported()) return false;
        bound::Field<N, Observations, DIMENSIONS> input;
        for (int node = 0; node < peano4::power(N, DIMENSIONS); node++) {
          std::array<double, Observations> magnitudes;
          for (int observation = 0; observation < Observations; observation++) {
            std::array<Interval, V> terms;
            for (int component = 0; component < V; component++) {
              const double weight = C::observableWeight(observation, component);
              terms[component] = bound::mul(residual[node * V + component], {weight, weight});
            }
            const auto value = bound::div(bound::canonicalSum<V>(terms), budget(observation));
            if (not bound::valid(value)) return false;
            input[node * Observations + observation] = value;
            magnitudes[observation] = bound::magnitude(value);
          }
          if (not bound::normPasses<Observations>(magnitudes, 1.0)) return false;
        }
        if (finiteVolume or N == 1) return true;
        if constexpr (requires { C::transferWeightLower(0, 0); C::transferWeightUpper(0, 0); }) {
          std::array<Interval, N * N> matrix;
          for (int node = 0; node < N; node++) for (int mode = 0; mode < N; mode++)
            matrix[mode * N + node] = {C::transferWeightLower(node, mode), C::transferWeightUpper(node, mode)};
          return bound::passesIntervals<N, Observations, DIMENSIONS>(input, 1.0, matrix);
        } else return false;
      }
    }

    /** The same check for a child (fine) and the values that come back from its parent (restored). FV compares
     * the averages. DG bounds the difference everywhere in the cell.
     */
    template <int N, typename P>
    GPU_CALLABLE_METHOD bool transferPasses(const P* fine, const P* restored, bool finiteVolume) const {
      if constexpr (Observations == 0) return true;
      else {
        namespace bound = detail::transfer_bound;
        if (not _valid or _empty or fine == nullptr or restored == nullptr or not bound::arithmeticSupported()) return false;
        bound::Field<N, Observations, DIMENSIONS> input;
        for (int node = 0; node < peano4::power(N, DIMENSIONS); node++) {
          std::array<Interval, V> differences;
          for (int component = 0; component < V; component++) {
            differences[component] = bound::difference(static_cast<double>(fine[node * V + component]),
              static_cast<double>(restored[node * V + component]), 1.0);
            if (not bound::valid(differences[component])) return false;
          }
          std::array<double, Observations> magnitudes;
          for (int observation = 0; observation < Observations; observation++) {
            std::array<Interval, V> terms;
            for (int component = 0; component < V; component++) {
              const double weight = C::observableWeight(observation, component);
              terms[component] = bound::mul(differences[component], {weight, weight});
            }
            const auto value = bound::div(bound::canonicalSum<V>(terms), budget(observation));
            if (not bound::valid(value)) return false;
            input[node * Observations + observation] = value;
            magnitudes[observation] = bound::magnitude(value);
          }
          if (not bound::normPasses<Observations>(magnitudes, 1.0)) return false;
        }
        if (finiteVolume or N == 1) return true;
        if constexpr (requires { C::transferWeightLower(0, 0); C::transferWeightUpper(0, 0); }) {
          std::array<Interval, N * N> matrix;
          for (int node = 0; node < N; node++) for (int mode = 0; mode < N; mode++) {
            matrix[mode * N + node] = {C::transferWeightLower(node, mode), C::transferWeightUpper(node, mode)};
          }
          return bound::passesIntervals<N, Observations, DIMENSIONS>(input, 1.0, matrix);
        } else return false;
      }
    }
  };

  /** Checks the observables when a DG cell is compared with the initial condition (initialDataPasses()). It
   * collects the initial condition and the polynomial of the cell on every child that a refinement would
   * create, and tests them with one tolerance per observable for all children.
   */
  template <int N, int V, typename C, typename P>
  class InitialDataMonitor {
    static constexpr int Values = peano4::power(N, DIMENSIONS) * V;
    Stencil<V, C> _stencil;
    std::vector<P> _fresh, _represented;
    int _children = 0;
    bool _valid = true;

  public:
    /** Adds one child: the initial condition at its nodes and the values of the cell's polynomial there. */
    bool add(const P* fresh, const P* represented) {
      if constexpr (count<C>() == 0) return true;
      else {
        if (not _valid or fresh == nullptr or represented == nullptr
            or _children >= peano4::ChildrenPerCell) return _valid = false;
        if (not _stencil.add(AcceptedCellView<P>{fresh, N, AcceptedRepresentation::DGNodal}))
          return _valid = false;
        if (_children == 0) {
          _fresh.reserve(static_cast<std::size_t>(Values) * peano4::ChildrenPerCell);
          _represented.reserve(static_cast<std::size_t>(Values) * peano4::ChildrenPerCell);
        }
        _fresh.insert(_fresh.end(), fresh, fresh + Values);
        _represented.insert(_represented.end(), represented, represented + Values);
        ++_children;
        return true;
      }
    }

    /** Do the observables stay within their tolerances on all children? */
    bool passes() const {
      if constexpr (count<C>() == 0) return true;
      else {
        if (not _valid or _children != peano4::ChildrenPerCell) return false;
        for (int child = 0; child < _children; child++) {
          if (not _stencil.template transferPasses<N>(
                _fresh.data() + child * Values, _represented.data() + child * Values, true)) return false;
        }
        return true;
      }
    }
  };

  TARGET_DECLARE
  /** InitialDataMonitor for a GPU kernel, with arrays of fixed size. */
  template <int N, int V, typename C, typename P>
  class DeviceInitialDataMonitor {
    static constexpr int Values = peano4::power(N, DIMENSIONS) * V;
    Stencil<V, C> _stencil;
    static constexpr int StoredValues = count<C>() == 0 ? 0 : Values * peano4::ChildrenPerCell;
    std::array<P, StoredValues> _fresh{}, _represented{};
    int _children = 0;
    bool _valid = true;

  public:
    GPU_CALLABLE_METHOD bool add(const P* fresh, const P* represented) {
      if constexpr (count<C>() == 0) return true;
      else {
        if (not _valid or fresh == nullptr or represented == nullptr
            or _children >= peano4::ChildrenPerCell) return _valid = false;
        if (not _stencil.add(AcceptedCellView<P>{fresh, N, AcceptedRepresentation::DGNodal}))
          return _valid = false;
        for (int entry = 0; entry < Values; entry++) {
          _fresh[_children * Values + entry] = fresh[entry];
          _represented[_children * Values + entry] = represented[entry];
        }
        ++_children;
        return true;
      }
    }

    GPU_CALLABLE_METHOD bool passes() const {
      if constexpr (count<C>() == 0) return true;
      else {
        if (not _valid or _children != peano4::ChildrenPerCell) return false;
        for (int child = 0; child < _children; child++) {
          if (not _stencil.template transferPasses<N>(
                _fresh.data() + child * Values, _represented.data() + child * Values, true)) return false;
        }
        return true;
      }
    }
  };

  TARGET_DECLARE_END

  /** Returns the prediction score of a face for the observables, in units of their tolerances. A result above
   * one means that an observable exceeds its tolerance. */
  template <int N, int V, typename C, typename P>
  double faceRatio(AcceptedCellView<P> own, AcceptedCellView<P> other,
                   const NativeCellKey& ownKey, const NativeCellKey& otherKey,
                   const double* nodes, int normal, int ownSide) {
    if constexpr (count<C>() == 0) return 0.0;
    else {
      Stencil<V, C> stencil;
      std::vector<double> left, right;
      if (not stencil.add(own) or not stencil.add(other)
          or not stencil.project(own, left) or not stencil.project(other, right)) return INFINITY;
      return prediction_detail::faceDetail<N, count<C>(), UnitConfig<C>>(
        AcceptedCellView<double>{left.data(), own.pointsPerAxis, own.representation},
        AcceptedCellView<double>{right.data(), other.pointsPerAxis, other.representation},
        ownKey, otherKey, nodes, normal, ownSide);
    }
  }

  /** Returns the score of one cell for the observables, in units of their tolerances. With initial set, the
   * cell is judged from its own values alone, as on the initial mesh. */
  template <int N, int V, typename C, typename P>
  double localRatio(AcceptedCellView<P> view, const double* nodes, bool initial) {
    if constexpr (count<C>() == 0) return 0.0;
    else {
      // A DG cell has no score of its own in prediction mode. Its faces carry it.
      if (not initial and not view.isFiniteVolume()) {
        return valid<V, C>() ? prediction_detail::localDetail<N, V, UnitConfig<C>>(view, nodes) : INFINITY;
      }
      Stencil<V, C> stencil;
      std::vector<double> values;
      if (not stencil.add(view) or not stencil.project(view, values)) return INFINITY;
      const AcceptedCellView<double> observed{values.data(), view.pointsPerAxis, view.representation};
      if (not initial) return prediction_detail::localDetail<N, count<C>(), UnitConfig<C>>(observed, nodes);
      if (view.isFiniteVolume()) return resolution_detail::fvResolutionDetail<count<C>(), UnitConfig<C>>(
        values.data(), view.pointsPerAxis, 1.0);
      if (view.pointsPerAxis != N) return INFINITY;
      return amr_detail::highestOrderDetailImpl<true, true, N, count<C>(), UnitConfig<C>>(values.data(), 1.0);
    }
  }
  TARGET_DECLARE
  /** faceRatio() for a GPU kernel. */
  template <int N, int V, typename C, int MaxPoints, typename P>
  GPU_CALLABLE_METHOD double deviceFaceRatio(
    AcceptedCellView<P> own, AcceptedCellView<P> other,
    const NativeCellKey& ownKey, const NativeCellKey& otherKey,
    const double* nodes, int normal, int ownSide
  ) {
    if constexpr (count<C>() == 0) return 0.0;
    else {
      if (own.pointsPerAxis > MaxPoints or other.pointsPerAxis > MaxPoints) return INFINITY;
      constexpr int Capacity = peano4::power(MaxPoints, DIMENSIONS) * count<C>();
      Stencil<V, C> stencil;
      std::array<double, Capacity> left, right;
      if (not stencil.add(own) or not stencil.add(other)
          or not stencil.project(own, left.data(), Capacity) or not stencil.project(other, right.data(), Capacity)) return INFINITY;
      return prediction_detail::deviceFaceDetail<N, count<C>(), UnitConfig<C>, MaxPoints>(
        AcceptedCellView<double>{left.data(), own.pointsPerAxis, own.representation},
        AcceptedCellView<double>{right.data(), other.pointsPerAxis, other.representation},
        ownKey, otherKey, nodes, normal, ownSide);
    }
  }

  /** localRatio() for a GPU kernel. */
  template <int N, int V, typename C, int MaxPoints, typename P>
  GPU_CALLABLE_METHOD double deviceLocalRatio(AcceptedCellView<P> view, const double* nodes, bool initial) {
    if constexpr (count<C>() == 0) return 0.0;
    else {
      if (view.pointsPerAxis > MaxPoints) return INFINITY;
      if (not initial and not view.isFiniteVolume()) {
        return valid<V, C>() ? prediction_detail::deviceLocalDetail<N, V, UnitConfig<C>, MaxPoints>(view, nodes) : INFINITY;
      }
      constexpr int Capacity = peano4::power(MaxPoints, DIMENSIONS) * count<C>();
      Stencil<V, C> stencil;
      std::array<double, Capacity> values;
      if (not stencil.add(view) or not stencil.project(view, values.data(), Capacity)) return INFINITY;
      const AcceptedCellView<double> observed{values.data(), view.pointsPerAxis, view.representation};
      if (not initial) return prediction_detail::deviceLocalDetail<N, count<C>(), UnitConfig<C>, MaxPoints>(observed, nodes);
      if (view.isFiniteVolume()) return resolution_detail::fvResolutionDetail<count<C>(), UnitConfig<C>>(
        values.data(), view.pointsPerAxis, 1.0);
      if (view.pointsPerAxis != N) return INFINITY;
      return amr_detail::highestOrderDetailImpl<true, true, N, count<C>(), UnitConfig<C>>(values.data(), 1.0);
    }
  }
  TARGET_DECLARE_END
} // namespace exahype2::amr::observation_detail
