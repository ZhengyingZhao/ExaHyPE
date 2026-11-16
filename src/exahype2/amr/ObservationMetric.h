// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "PredictionSensor.h"

namespace exahype2::amr::observation_detail {
  /** Return the number of optional linear observation channels. */
  template <typename C>
  constexpr int count() {
    if constexpr (requires { C::NumberOfObservables; }) return C::NumberOfObservables;
    return 0;
  }

  template <int V, typename C>
  bool valid() {
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

  /** Unit tolerances for observations normalized by their physical budgets. */
  template <typename C>
  struct UnitConfig {
    static double stateScale(int) { return 1.0; }
    static constexpr double ResolutionTolerance = 1.0;
    static constexpr double TransferTolerance = 1.0;
    static double highestModeWeight(int node) requires requires { C::highestModeWeight(node); } {
      return C::highestModeWeight(node);
    }
  };

  /**
   * Fix observation budgets from the accepted stencil:
   *   absolute tolerance + relative tolerance * observed nodal range.
   * Projection and trial parents cannot enlarge the budget. Linear observations
   * preserve DG degree and commute with FV averaging.
   */
  template <int V, typename C>
  class Stencil {
    static constexpr int Observations = count<C>();
    static_assert(V > 0 and Observations >= 0);
    using Interval = detail::transfer_bound::Interval;
    std::array<double, V> _lower{}, _upper{};
    // A lower bound on the nodal range prevents rounding from enlarging budgets.
    std::array<double, Observations> _minimumUpper{}, _maximumLower{};
    bool _valid = observation_detail::valid<V, C>();
    bool _empty = true;

    template <typename P>
    Interval observationInterval(const P* state, int observation) const {
      namespace bound = detail::transfer_bound;
      std::array<Interval, V> terms;
      for (int component = 0; component < V; component++) {
        const double weight = C::observableWeight(observation, component);
        const double value = static_cast<double>(state[component]);
        terms[component] = bound::mul({weight, weight}, {value, value});
      }
      return bound::canonicalSum<V>(terms);
    }

  public:
    Stencil() {
      _lower.fill(INFINITY);
      _upper.fill(-INFINITY);
      _minimumUpper.fill(INFINITY);
      _maximumLower.fill(-INFINITY);
    }

    template <typename P>
    bool add(AcceptedCellView<P> view) {
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

    /** Return a positive budget, or zero when its bound is invalid. */
    double budget(int observation) const {
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

    /**
     * Normalize linear observations after centering each component at its midrange.
     * Centering preserves weak differences against a large constant background.
     */
    template <typename P>
    bool project(AcceptedCellView<P> view, std::vector<double>& output) const {
      if constexpr (Observations == 0) { output.clear(); return true; }
      else {
        if (not _valid or _empty or view.values == nullptr or view.pointsPerAxis <= 0) return false;
        int states = 1;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          if (states > std::numeric_limits<int>::max() / view.pointsPerAxis) return false;
          states *= view.pointsPerAxis;
        }
        if (states > std::numeric_limits<int>::max() / std::max(V, Observations)) return false;
        std::array<double, Observations> budgets;
        for (int observation = 0; observation < Observations; observation++) {
          budgets[observation] = budget(observation);
          if (not std::isfinite(budgets[observation]) or budgets[observation] <= 0.0) return false;
        }
        output.resize(static_cast<std::size_t>(states) * Observations);
        for (int node = 0; node < states; node++) {
          std::array<double, V> differences;
          for (int component = 0; component < V; component++) {
            const double reference = 0.5 * _lower[component] + 0.5 * _upper[component];
            differences[component] = native_neighbour_detail::scaledDifference(
              static_cast<double>(view.values[node * V + component]), reference, 1.0);
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
    }

    /** Check fine-minus-parent observation residuals with the fine-family budget.
     * Input intervals enclose unscaled residuals at fine nodes; DG also requires
     * a continuous polynomial bound.
     */
    template <int N>
    bool transferIntervalsPasses(const detail::transfer_bound::Field<N, V, DIMENSIONS>& residual, bool finiteVolume) const {
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

    /**
     * Check fine-minus-restored observations with the accepted stencil's budget.
     * Combine raw residual intervals before normalization to retain uncertainty.
     * FV checks cell means; DG also requires a continuous polynomial bound.
     */
    template <int N, typename P>
    bool transferPasses(const P* fine, const P* restored, bool finiteVolume) const {
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

  /**
   * Check supplied initial-data samples using observation budgets from the
   * complete fine sample family.
   */
  template <int N, int V, typename C, typename P>
  class InitialDataMonitor {
    static constexpr int Values = peano4::power(N, DIMENSIONS) * V;
    Stencil<V, C> _stencil;
    std::vector<P> _fresh, _represented;
    int _children = 0;
    bool _valid = true;

  public:
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

    bool passes() const {
      if constexpr (count<C>() == 0) return true;
      else {
        if (not _valid or _children != peano4::ChildrenPerCell) return false;
        for (int child = 0; child < _children; child++) {
          // Use the pointwise check for initial-data samples.
          if (not _stencil.template transferPasses<N>(
                _fresh.data() + child * Values, _represented.data() + child * Values, true)) return false;
        }
        return true;
      }
    }
  };

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

  template <int N, int V, typename C, typename P>
  double localRatio(AcceptedCellView<P> view, const double* nodes, bool initial) {
    if constexpr (count<C>() == 0) return 0.0;
    else {
      // A single DG cell has zero linear-observation prediction residual.
      // Validate the view without constructing observation fields.
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
} // namespace exahype2::amr::observation_detail
