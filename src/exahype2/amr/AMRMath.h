// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "TransferBounds.h"
#include "NativeNeighbours.h"
#include "exahype2/dg/Basis.h"
#include "peano4/Loop.h"
#include "tarch/la/ScalarOperations.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

// Numerical helpers of the AMR criterion: the transfer of a cell between a parent and its children, the test
// of the transfer error, the score of the highest polynomial degree, and the check of a DG cell against the
// initial condition. A helper returns infinity or false for input that is not finite or would overflow.
namespace exahype2::amr {
  namespace amr_detail {
    namespace transfer_bound = ::exahype2::amr::detail::transfer_bound;
    TARGET_DECLARE
    /** Returns the resolution tolerance of Config, or its transfer tolerance if it sets none. */
    template <typename C>
    GPU_CALLABLE_METHOD constexpr double resolutionTolerance() {
      if constexpr (requires { C::ResolutionTolerance; }) return C::ResolutionTolerance;
      return C::TransferTolerance;
    }
    /** Does this value count for the score in prediction mode? Config can select values. Without a selection
     * all of them count. */
    template <typename C>
    GPU_CALLABLE_METHOD bool resolutionComponent(int component) {
      if constexpr (requires { C::resolutionComponent(component); }) return C::resolutionComponent(component);
      return true;
    }
    /** Adds one step to the quiet time of a cell, measured in the time a wave takes to cross the cell.
     * crossing is that time. A cell in which nothing moves uses referenceTime instead. Returns -1 for an
     * invalid crossing time. */
    GPU_CALLABLE_METHOD inline double advanceQuiet(double quiet, double crossing, bool moving, double dt, double referenceTime) {
      if (not moving) crossing = referenceTime;
      if (crossing <= 0.0 or std::isnan(crossing)) return -1.0;
      const double limit = std::numeric_limits<double>::max();
      const double increment = crossing < 1.0 and dt > limit * crossing ? limit : dt / crossing;
      return quiet > limit - increment ? limit : quiet + increment;
    }

    /** Returns sqrt(left^2 + right^2), or infinity if that overflows or an argument is not finite. */
    GPU_CALLABLE_METHOD inline double guardedHypot(double left, double right) {
      const double limit = std::numeric_limits<double>::max();
      const double invalid = std::numeric_limits<double>::infinity();
      if (not std::isfinite(left) or not std::isfinite(right)) return invalid;
      left = std::abs(left);
      right = std::abs(right);
      if (left <= limit * 0.5 and right <= limit * 0.5) return std::hypot(left, right);
      const double largest = std::max(left, right), smallest = std::min(left, right);
      if (smallest == 0.0) return largest;
      const double ratio = smallest / largest;
      const double factor = std::sqrt(1.0 + ratio * ratio);
      if (largest > std::nextafter(limit / factor, 0.0)) return invalid;
      // volatile keeps the compiler from computing the product before the overflow test above.
      volatile double boundedLargest = largest;
      return boundedLargest * factor;
    }

    /** Evaluate the parent's DG polynomial at the nodes of one child. */
    template <int N, int V, typename P>
    GPU_CALLABLE_METHOD void interpolateDGCell(const P* parent, P* child, const double* matrix, int position) {
      double prepared[exahype2::dg::ReferencesAndBoundsEntries<V>];
      for (int component = 0; component < V; component++) {
        exahype2::dg::computeReferenceAndBoundOfParent<N, V>(parent, matrix, component, prepared);
      }
      for (int node = 0; node < peano4::power(N, DIMENSIONS); node++) {
        exahype2::dg::interpolateNode<N, V>(parent, child, matrix, prepared, position, node);
      }
    }

    /** Project the children's DG solutions onto their parent. */
    template <int N, int V, typename P>
    GPU_CALLABLE_METHOD void restrictDGCell(const P* const* children, P* parent, const double* matrix) {
      double prepared[exahype2::dg::ReferencesAndBoundsEntries<V>];
      for (int component = 0; component < V; component++) {
        exahype2::dg::computeReferenceAndBoundOfChildren<N, V>(children, matrix, component, prepared);
      }
      for (int node = 0; node < peano4::power(N, DIMENSIONS); node++) {
        exahype2::dg::restrictNode<N, V>(children, parent, matrix, prepared, node);
      }
    }

    /** Give each fine volume the average of the coarse volume that contains it. */
    template <int N, int V, typename P>
    GPU_CALLABLE_METHOD void interpolateVolumes(const P* parent, P* child, int childPosition) {
      const auto position = peano4::dDelinearised(childPosition, peano4::SubdivisionFactor);
      for (int volume = 0; volume < peano4::power(N, DIMENSIONS); volume++) {
        const auto fine = peano4::dDelinearised(volume, N);
        tarch::la::Vector<DIMENSIONS, int> coarse;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          coarse(axis) = (fine(axis) + N * position(axis)) / peano4::SubdivisionFactor;
        }
        const int fineIndex = peano4::dLinearised(fine, N) * V;
        const int coarseIndex = peano4::dLinearised(coarse, N) * V;
        for (int component = 0; component < V; component++) child[fineIndex + component] = parent[coarseIndex + component];
      }
    }

    /** Average the fine volumes that make up each coarse volume. */
    template <int N, int V, typename P>
    GPU_CALLABLE_METHOD void restrictVolumes(const P* const* children, P* parent) {
      for (int volume = 0; volume < peano4::power(N, DIMENSIONS); volume++) {
        const auto coarse = peano4::dDelinearised(volume, N);
        const int target = peano4::dLinearised(coarse, N) * V;
        for (int component = 0; component < V; component++) {
          double values[peano4::ChildrenPerCell];
          for (int subvolume = 0; subvolume < peano4::ChildrenPerCell; subvolume++) {
            const auto offset = peano4::dDelinearised(subvolume, peano4::SubdivisionFactor);
            tarch::la::Vector<DIMENSIONS, int> position, fine;
            for (int axis = 0; axis < DIMENSIONS; axis++) {
              const int global = peano4::SubdivisionFactor * coarse(axis) + offset(axis);
              position(axis) = global / N;
              fine(axis) = global % N;
            }
            const P* child = children[peano4::dLinearised(position, peano4::SubdivisionFactor)];
            values[peano4::dLinearised(offset, peano4::SubdivisionFactor)] =
              static_cast<double>(child[peano4::dLinearised(fine, N) * V + component]);
          }
          parent[target + component] = static_cast<P>(tarch::la::meanInFixedPoint(values));
        }
      }
    }

    /** Can a cell be projected with this matrix, axis by axis, without leaving the range of its storage
     * type? The test bounds the intermediate values by the largest input and the largest row sum. */
    template <bool NativeArithmetic, int Inputs, int Samples, int V, typename P, typename MatrixPrecision>
    GPU_CALLABLE_METHOD bool projectionIsSafe(const P* Q, const MatrixPrecision* matrix) {
      if (Q == nullptr or matrix == nullptr) return false;
      const double limit = static_cast<double>(std::numeric_limits<P>::max());
      double bound = 0.0;
      for (int entry = 0; entry < peano4::power(Inputs, DIMENSIONS) * V; entry++) {
        const double value = static_cast<double>(Q[entry]);
        if (not std::isfinite(value)) return false;
        bound = std::max(bound, std::abs(value));
      }
      double rowBound = 0.0;
      for (int sample = 0; sample < Samples; sample++) {
        double sum = 0.0;
        for (int input = 0; input < Inputs; input++) {
          const double value = std::abs(static_cast<double>(matrix[input * Samples + sample]));
          if (not std::isfinite(value) or sum > limit - value) return false;
          sum += value;
        }
        rowBound = std::max(rowBound, sum);
      }
      // Each axis can enlarge the bound by the row sum. A factor of two covers the rounding.
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        // The projection first adds mirrored entries, so twice the bound has to fit as well.
        if constexpr (Inputs > 1) if (bound > limit / 2.0) return false;
        if constexpr (NativeArithmetic) {
          if (not native_neighbour_detail::product(bound, rowBound, bound)) return false;
          if (bound > limit / 2.0) return false;
          if (not native_neighbour_detail::product(bound, 2.0, bound)) return false;
        } else {
          if (rowBound > 1.0 and bound > limit / rowBound) return false;
          bound *= rowBound;
          if (bound > limit / 2.0) return false;
          bound *= 2.0;
        }
      }
      return true;
    }

    /** Returns (a-b)/scale. If a-b overflows, it divides first. Returns infinity if the result has no
     * finite value. */
    template <bool NativeArithmetic>
    GPU_CALLABLE_METHOD inline double scaledDifference(double a, double b, double scale) {
      if constexpr (NativeArithmetic) {
        return native_neighbour_detail::scaledDifference(a, b, scale);
      } else {
      const double limit = std::numeric_limits<double>::max();
      const double invalid = std::numeric_limits<double>::infinity();
      if (not std::isfinite(a) or not std::isfinite(b)) return invalid;
      if ((a > 0.0 and b < 0.0 and a > limit + b)
          or (a < 0.0 and b > 0.0 and a < -limit + b)) {
        if (scale <= 1.0) return invalid;
        a /= scale;
        b /= scale;
        if ((a > 0.0 and b < 0.0 and a > limit + b)
            or (a < 0.0 and b > 0.0 and a < -limit + b)) return invalid;
        return a - b;
      }
      const double delta = a - b;
      if (scale < 1.0 and std::abs(delta) > limit * scale) return invalid;
      return delta / scale;
      }
    }

    /** Does a polynomial stay within the transfer tolerance everywhere in the cell? input bounds the scaled
     * difference between a child and its parent at every node. */
    template <int N, int V, typename C>
    GPU_CALLABLE_METHOD bool transferPolynomialIntervalsPasses(const transfer_bound::Field<N, V, DIMENSIONS>& input) {
      if (not transfer_bound::arithmeticSupported()) return false;
      if constexpr (requires { C::transferWeightLower(0, 0); C::transferWeightUpper(0, 0); }) {
        std::array<transfer_bound::Interval, N * N> weights;
        for (int node = 0; node < N; node++) for (int mode = 0; mode < N; mode++)
          weights[mode * N + node] = {C::transferWeightLower(node, mode), C::transferWeightUpper(node, mode)};
        return transfer_bound::passesIntervals<N, V, DIMENSIONS>(input, C::TransferTolerance, weights);
      } else if constexpr (N == 1) {
        std::array<double, V> magnitudes;
        for (int component = 0; component < V; component++) magnitudes[component] = transfer_bound::magnitude(input[component]);
        return transfer_bound::normPasses<V>(magnitudes, C::TransferTolerance);
      } else return false;
    }

    /** A quick form of transferPolynomialIntervalsPasses(). It only knows an upper bound of the difference
     * per value. If it returns true, the full test passes as well. If it returns false, the full test decides.
     */
    template <int N, int V, typename C>
    GPU_CALLABLE_METHOD bool transferMagnitudesPass(const std::array<double, V>& magnitudes) {
      if (not transfer_bound::arithmeticSupported()) return false;
      const double budget = C::TransferTolerance;
      if (not std::isfinite(budget) or budget <= 0.0) return false;
      // A difference above the tolerance at a node fails the full test too.
      for (const double magnitude : magnitudes) if (not std::isfinite(magnitude) or magnitude > budget) return false;
      if constexpr (requires { C::transferWeightLower(0, 0); C::transferWeightUpper(0, 0); }) {
        std::array<transfer_bound::Interval, N * N> weights;
        for (int node = 0; node < N; node++) for (int mode = 0; mode < N; mode++)
          weights[mode * N + node] = {C::transferWeightLower(node, mode), C::transferWeightUpper(node, mode)};
        double amplification = 0.0;
        return transfer_bound::transformAmplification<N>(weights, amplification)
          and transfer_bound::amplifiedMaximaPass<V, DIMENSIONS>(magnitudes, budget, amplification);
      } else if constexpr (N == 1) {
        return transfer_bound::normPasses<V>(magnitudes, budget);
      } else return false;
    }

    /** Does a child differ from the polynomial that comes back from its parent by at most the transfer
     * tolerance, everywhere in the cell? */
    template <int N, int V, typename C, typename P>
    GPU_CALLABLE_METHOD bool transferPolynomialPasses(const P* fine, const P* restored) {
      if (fine == nullptr or restored == nullptr or not transfer_bound::arithmeticSupported()) return false;
      if constexpr (requires { C::transferWeightLower(0, 0); C::transferWeightUpper(0, 0); }) {
        std::array<transfer_bound::Interval, N * N> weights;
        std::array<double, V> scales;
        for (int node = 0; node < N; node++) {
          for (int mode = 0; mode < N; mode++) {
            weights[mode * N + node] = {C::transferWeightLower(node, mode), C::transferWeightUpper(node, mode)};
          }
        }
        for (int entry = 0; entry < V; entry++) scales[entry] = C::stateScale(entry);
        return transfer_bound::passes<N, V, DIMENSIONS>(fine, restored, scales, C::TransferTolerance, weights);
      } else if constexpr (N == 1) {
        // A constant has one value, so its one node decides.
        std::array<double, V> residuals;
        for (int entry = 0; entry < V; entry++) {
          residuals[entry] = transfer_bound::magnitude(transfer_bound::difference(
            fine[entry], restored[entry], C::stateScale(entry)));
        }
        return transfer_bound::normPasses<V>(residuals, C::TransferTolerance);
      } else {
        // Without the weights of Config there is no bound between the nodes.
        return false;
      }
    }
  /** Returns the largest scaled coefficient of the highest polynomial degree over all lines of nodes along
   * the axes. A large value means that the polynomial of the cell barely resolves the solution. With
   * StopAtBudget the search ends at the first value above budget. */
  template <bool NativeArithmetic, bool StopAtBudget, int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD double highestOrderDetailImpl(const P* Q, double budget) {
    static_assert(N > 0 and V > 0);
    if constexpr (N == 1 or not requires { C::highestModeWeight(0); }) {
      return 0.0;
    } else {
      const double invalid = std::numeric_limits<double>::infinity();
      const double limit = std::numeric_limits<double>::max();
      if (Q == nullptr) return invalid;
      std::array<double, N> weights;
      std::array<double, V> scales;
      for (int node = 0; node < N; node++) {
        weights[node] = C::highestModeWeight(node);
        if (not std::isfinite(weights[node]) or weights[node] == 0.0
            or (weights[node] > 0.0) != ((N - 1 - node) % 2 == 0)) return invalid;
      }
      for (int component = 0; component < V; component++) {
        scales[component] = C::stateScale(component);
        if (not std::isfinite(scales[component]) or scales[component] <= 0.0) return invalid;
      }
      // Mirrored nodes have the same weight up to its sign. One weight per pair serves every line and value.
      std::array<double, N / 2> pairWeights;
      for (int node = 0; node < N / 2; node++) {
        const int mirror = N - 1 - node;
        constexpr double parity = N % 2 == 0 ? -1.0 : 1.0;
        const double reflected = parity * weights[mirror];
        pairWeights[node] = weights[node] == reflected ? weights[node]
          : 0.5 * weights[node] + 0.5 * reflected;
        if (not std::isfinite(pairWeights[node]) or pairWeights[node] == 0.0) return invalid;
      }

      constexpr int Nodes = peano4::power(N, DIMENSIONS);
      double largest = 0.0;
      int stride = 1;
      for (int axis = 0; axis < DIMENSIONS; axis++, stride *= N) {
        for (int fibre = 0; fibre < Nodes / N; fibre++) {
          const int first = (fibre / stride) * (N * stride) + fibre % stride;
          std::array<double, V> magnitudes;
          for (int component = 0; component < V; component++) {
            std::array<double, N / 2> terms;
            for (int node = 0; node < N / 2; node++) {
              const int mirror = N - 1 - node;
              const double weight = pairWeights[node];
              const double left = static_cast<double>(Q[(first + node * stride) * V + component]);
              const double right = static_cast<double>(Q[(first + mirror * stride) * V + component]);
              double delta;
              if constexpr (N % 2 == 0) {
                delta = amr_detail::scaledDifference<NativeArithmetic>(left, right, scales[component]);
              } else {
                const double centre = static_cast<double>(Q[(first + (N / 2) * stride) * V + component]);
                const double a = amr_detail::scaledDifference<NativeArithmetic>(left, centre, scales[component]);
                const double b = amr_detail::scaledDifference<NativeArithmetic>(right, centre, scales[component]);
                if (not std::isfinite(a) or not std::isfinite(b)
                    or (a > 0.0 and b > 0.0 and a > limit - b)
                    or (a < 0.0 and b < 0.0 and a < -limit - b)) return invalid;
                delta = a + b;
              }
              if (not std::isfinite(delta)
                  or (std::abs(weight) > 1.0 and std::abs(delta) > limit / std::abs(weight))
                  or (std::abs(delta) > 1.0 and std::abs(weight) > limit / std::abs(delta))) return invalid;
              terms[node] = weight * delta;
            }
            transfer_bound::sort(terms.begin(), terms.end());
            double positive = 0.0, negative = 0.0;
            for (const double term : terms) {
              if (term >= 0.0) {
                if (positive > limit - term) return invalid;
                positive += term;
              }
            }
            for (int entry = int(terms.size()); entry-- > 0;) {
              if (terms[entry] < 0.0) {
                if (negative > limit + terms[entry]) return invalid;
                negative += -terms[entry];
              }
            }
            magnitudes[component] = std::abs(positive - negative);
          }
          transfer_bound::sort(magnitudes.begin(), magnitudes.end());
          double norm = 0.0;
          for (const double magnitude : magnitudes) {
            if constexpr (NativeArithmetic) norm = amr_detail::guardedHypot(norm, magnitude);
            else norm = std::hypot(norm, magnitude);
          }
          if (not std::isfinite(norm)) return invalid;
          largest = std::max(largest, norm);
          if constexpr (StopAtBudget) if (largest > budget) return largest;
        }
      }
      return largest;
    }
  }

  /** Stands in for the check of the observables where Config has none. Every sample passes. */
  struct NoInitialObservation {
    template <typename P>
    GPU_CALLABLE_METHOD bool add(const P*, const P*) { return true; }
    GPU_CALLABLE_METHOD bool passes() const { return true; }
  };

  /** Checks the input of initialDataPasses() and returns the width of a child cell in childWidth. */
  template <bool NativeArithmetic, int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD bool prepareInitialSampling(
    const P* Q, const double* nodes, const double* prolongation,
    const tarch::la::Vector<DIMENSIONS, double>& x, const tarch::la::Vector<DIMENSIONS, double>& h,
    tarch::la::Vector<DIMENSIONS, double>& childWidth
  ) {
    if (not transfer_bound::arithmeticSupported()
        or not std::isfinite(resolutionTolerance<C>()) or resolutionTolerance<C>() <= 0.0) return false;
    for (int node = 0; node < N; node++) {
      if (not std::isfinite(nodes[node]) or nodes[node] < 0.0 or nodes[node] > 1.0
          or (node > 0 and nodes[node] <= nodes[node - 1])) return false;
    }
    for (int interval = 0; interval < peano4::SubdivisionFactor; interval++) {
      if (not amr_detail::projectionIsSafe<NativeArithmetic, N, N, V>(Q, prolongation + interval * N * N)) return false;
    }
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (not std::isfinite(x(axis)) or not std::isfinite(h(axis)) or h(axis) <= 0.0) return false;
      childWidth(axis) = h(axis) / peano4::SubdivisionFactor;
      if (not std::isfinite(childWidth(axis)) or childWidth(axis) <= 0.0) return false;
    }
    return true;
  }

  /** Does the polynomial of a cell agree with the initial condition at one point, within the resolution
   * tolerance? fresh is the initial condition there, represented the value of the polynomial. */
  template <int V, typename C, typename P>
  GPU_CALLABLE_METHOD bool initialSampleResidualPasses(const P* fresh, const P* represented) {
    std::array<double, V> residuals;
    for (int component = 0; component < V; component++) {
      if constexpr (requires { C::resolutionComponent(component); }) {
        if (not std::isfinite(static_cast<double>(fresh[component]))
            or not std::isfinite(static_cast<double>(represented[component]))) return false;
        if (not resolutionComponent<C>(component)) { residuals[component] = 0.0; continue; }
      }
      residuals[component] = transfer_bound::magnitude(transfer_bound::difference(
        static_cast<double>(fresh[component]), static_cast<double>(represented[component]), C::stateScale(component)
      ));
    }
    return transfer_bound::normPasses<V>(residuals, resolutionTolerance<C>());
  }

  TARGET_DECLARE_END

  /** Does a DG cell represent the initial condition well enough? The initial condition is evaluated at the
   * nodes of the cells that a refinement would create and compared with the polynomial of the cell there. A
   * cell that fails refines. */
  template <bool NativeArithmetic, int N, int V, typename C, typename P,
            typename Observer = NoInitialObservation>
  bool initialDataPasses(
    const P* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const double* nodes, const double* prolongation,
    std::type_identity_t<std::function<void(P*, const tarch::la::Vector<DIMENSIONS, double>&,
                                          const tarch::la::Vector<DIMENSIONS, double>&)>> initialCondition,
    std::type_identity_t<std::function<bool(const P*, int, const tarch::la::Vector<DIMENSIONS, double>&,
                                          const tarch::la::Vector<DIMENSIONS, double>&)>> admissible,
    Observer observer = {}
  ) {
    static_assert(N > 0 and V > 0);
    constexpr int Nodes = peano4::power(N, DIMENSIONS), Values = Nodes * V;
    using Coordinates = tarch::la::Vector<DIMENSIONS, double>;
    if (Q == nullptr or nodes == nullptr or prolongation == nullptr or initialCondition == nullptr) return false;
    Coordinates childWidth;
    if (not prepareInitialSampling<NativeArithmetic, N, V, C>(Q, nodes, prolongation, x, h, childWidth)) return false;
    for (int child = 0; child < peano4::ChildrenPerCell; child++) {
      Coordinates centre;
      const auto position = peano4::dDelinearised(child, peano4::SubdivisionFactor);
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        centre(axis) = x(axis) + (position(axis) + 0.5 - 0.5 * peano4::SubdivisionFactor) * childWidth(axis);
        if (not std::isfinite(centre(axis))) return false;
      }
      P fresh[Values], represented[Values];
      std::fill_n(fresh, Values, std::numeric_limits<P>::quiet_NaN());
      amr_detail::interpolateDGCell<N, V>(Q, represented, prolongation, child);
      for (int node = 0; node < Nodes; node++) {
        const auto point = exahype2::dg::getQuadraturePoint(centre, childWidth, peano4::dDelinearised(node, N), nodes);
        for (int axis = 0; axis < DIMENSIONS; axis++) if (not std::isfinite(point(axis))) return false;
        initialCondition(fresh + node * V, point, childWidth);
        if (not initialSampleResidualPasses<V, C>(fresh + node * V, represented + node * V)) return false;
      }
      if (admissible != nullptr and (not admissible(fresh, Nodes, centre, childWidth)
          or not admissible(represented, Nodes, centre, childWidth))) return false;
      if (not observer.add(fresh, represented)) return false;
    }
    return observer.passes();
  }

  TARGET_DECLARE
  /** Is the callback set? A callback that cannot be compared with nullptr always is. */
  template <typename Callback>
  GPU_CALLABLE_METHOD bool callbackAvailable(const Callback& callback) {
    if constexpr (requires { callback != nullptr; }) return callback != nullptr;
    else return true;
  }

  /** The check of initialDataPasses() with callbacks that a GPU kernel can call. */
  template <bool NativeArithmetic, int N, int V, typename C, typename P,
            typename InitialCondition, typename Admissible, typename Observer = NoInitialObservation>
  GPU_CALLABLE_METHOD bool deviceInitialDataPasses(
    const P* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const double* nodes, const double* prolongation,
    InitialCondition initialCondition, Admissible admissible,
    Observer observer = {}
  ) {
    if constexpr (std::is_same_v<InitialCondition, std::nullptr_t>) return false;
    else {
      static_assert(N > 0 and V > 0);
      constexpr int Nodes = peano4::power(N, DIMENSIONS), Values = Nodes * V;
      using Coordinates = tarch::la::Vector<DIMENSIONS, double>;
      if (Q == nullptr or nodes == nullptr or prolongation == nullptr or not callbackAvailable(initialCondition)) return false;
      Coordinates childWidth;
      if (not prepareInitialSampling<NativeArithmetic, N, V, C>(Q, nodes, prolongation, x, h, childWidth)) return false;
      for (int child = 0; child < peano4::ChildrenPerCell; child++) {
        Coordinates centre;
        const auto position = peano4::dDelinearised(child, peano4::SubdivisionFactor);
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          centre(axis) = x(axis) + (position(axis) + 0.5 - 0.5 * peano4::SubdivisionFactor) * childWidth(axis);
          if (not std::isfinite(centre(axis))) return false;
        }
        P fresh[Values], represented[Values];
        for (int value = 0; value < Values; value++) fresh[value] = std::numeric_limits<P>::quiet_NaN();
        amr_detail::interpolateDGCell<N, V>(Q, represented, prolongation, child);
        for (int node = 0; node < Nodes; node++) {
          const auto point = exahype2::dg::getQuadraturePoint(centre, childWidth, peano4::dDelinearised(node, N), nodes);
          for (int axis = 0; axis < DIMENSIONS; axis++) if (not std::isfinite(point(axis))) return false;
          initialCondition(fresh + node * V, point, childWidth);
          if (not initialSampleResidualPasses<V, C>(fresh + node * V, represented + node * V)) return false;
        }
        if constexpr (not std::is_same_v<Admissible, std::nullptr_t>) {
          if (callbackAvailable(admissible) and (not admissible(fresh, Nodes, centre, childWidth)
              or not admissible(represented, Nodes, centre, childWidth))) return false;
        }
        if (not observer.add(fresh, represented)) return false;
      }
      return observer.passes();
    }
  }

  TARGET_DECLARE_END

  } // namespace amr_detail

  // The helpers above under the names that the two criteria use: native_detail for a solver without a
  // limiter (NativeArithmetic set), detail for ADER-DG with the limiter (NativeArithmetic not set).
  TARGET_DECLARE
  namespace native_detail {
    namespace transfer_bound = ::exahype2::amr::detail::transfer_bound;
    using amr_detail::guardedHypot;
    using amr_detail::interpolateVolumes;
    using amr_detail::restrictVolumes;
    using amr_detail::transferPolynomialPasses;
    template <int I, int S, int V, typename P, typename M>
    GPU_CALLABLE_METHOD bool projectionIsSafe(const P* Q, const M* matrix) {
      return amr_detail::projectionIsSafe<true, I, S, V>(Q, matrix);
    }
    GPU_CALLABLE_METHOD inline double scaledDifference(double a, double b, double scale) {
      return amr_detail::scaledDifference<true>(a, b, scale);
    }
    template <bool Stop, int N, int V, typename C, typename P>
    GPU_CALLABLE_METHOD double nativeHighestOrderDetailImpl(const P* Q, double budget) {
      return amr_detail::highestOrderDetailImpl<true, Stop, N, V, C>(Q, budget);
    }
  }
  namespace detail {
    using amr_detail::transferPolynomialPasses;
    template <int I, int S, int V, typename P, typename M>
    GPU_CALLABLE_METHOD bool projectionIsSafe(const P* Q, const M* matrix) {
      return amr_detail::projectionIsSafe<false, I, S, V>(Q, matrix);
    }
    GPU_CALLABLE_METHOD inline double scaledDifference(double a, double b, double scale) {
      return amr_detail::scaledDifference<false>(a, b, scale);
    }
    template <bool Stop, int N, int V, typename C, typename P>
    GPU_CALLABLE_METHOD double highestOrderDetailImpl(const P* Q, double budget) {
      return amr_detail::highestOrderDetailImpl<false, Stop, N, V, C>(Q, budget);
    }
  }
  /** The score of the highest polynomial degree of one cell, for a solver without a limiter. */
  template <int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD double nativeHighestOrderDetail(const P* Q) {
    return amr_detail::highestOrderDetailImpl<true, false, N, V, C>(Q, std::numeric_limits<double>::infinity());
  }
  /** The score of the highest polynomial degree of one cell, for ADER-DG with the limiter. */
  template <int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD double highestOrderDetail(const P* Q) {
    return amr_detail::highestOrderDetailImpl<false, false, N, V, C>(Q, std::numeric_limits<double>::infinity());
  }

  TARGET_DECLARE_END

  /** The check of a DG cell against the initial condition, for a solver without a limiter on the CPU. */
  template <int N, int V, typename C, typename P,
            typename Observer = amr_detail::NoInitialObservation>
  bool nativeInitialDataPasses(
    const P* Q, const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h, const double* nodes, const double* prolongation,
    std::type_identity_t<std::function<void(P*, const tarch::la::Vector<DIMENSIONS, double>&, const tarch::la::Vector<DIMENSIONS, double>&)>> initialCondition,
    std::type_identity_t<std::function<bool(const P*, int, const tarch::la::Vector<DIMENSIONS, double>&, const tarch::la::Vector<DIMENSIONS, double>&)>> admissible = nullptr,
    Observer observer = {}
  ) {
    return amr_detail::initialDataPasses<true, N, V, C>(
      Q, x, h, nodes, prolongation, initialCondition, admissible, std::move(observer));
  }

  /** The check of a DG cell against the initial condition, for ADER-DG with the limiter on the CPU. */
  template <int N, int V, typename C, typename P,
            typename Observer = amr_detail::NoInitialObservation>
  bool initialDataPasses(
    const P* Q, const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h, const double* nodes, const double* prolongation,
    std::type_identity_t<std::function<void(P*, const tarch::la::Vector<DIMENSIONS, double>&, const tarch::la::Vector<DIMENSIONS, double>&)>> initialCondition,
    std::type_identity_t<std::function<bool(const P*, int, const tarch::la::Vector<DIMENSIONS, double>&, const tarch::la::Vector<DIMENSIONS, double>&)>> admissible = nullptr,
    Observer observer = {}
  ) {
    return amr_detail::initialDataPasses<false, N, V, C>(
      Q, x, h, nodes, prolongation, initialCondition, admissible, std::move(observer));
  }
  TARGET_DECLARE
  /** The check of a DG cell against the initial condition, for a solver without a limiter in a GPU kernel. */
  template <int N, int V, typename C, typename P,
            typename InitialCondition, typename Admissible, typename Observer = amr_detail::NoInitialObservation>
  GPU_CALLABLE_METHOD bool nativeDeviceInitialDataPasses(
    const P* Q, const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h, const double* nodes, const double* prolongation,
    InitialCondition initialCondition, Admissible admissible, Observer observer = {}
  ) {
    return amr_detail::deviceInitialDataPasses<true, N, V, C>(
      Q, x, h, nodes, prolongation, initialCondition, admissible, observer);
  }

  /** The check of a DG cell against the initial condition, for ADER-DG with the limiter in a GPU kernel. */
  template <int N, int V, typename C, typename P,
            typename InitialCondition, typename Admissible, typename Observer = amr_detail::NoInitialObservation>
  GPU_CALLABLE_METHOD bool deviceInitialDataPasses(
    const P* Q, const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h, const double* nodes, const double* prolongation,
    InitialCondition initialCondition, Admissible admissible, Observer observer = {}
  ) {
    return amr_detail::deviceInitialDataPasses<false, N, V, C>(
      Q, x, h, nodes, prolongation, initialCondition, admissible, observer);
  }
  TARGET_DECLARE_END
} // namespace exahype2::amr
