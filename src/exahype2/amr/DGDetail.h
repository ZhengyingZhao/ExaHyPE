// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "NativeNeighbours.h"

namespace exahype2::amr::resolution_detail {
  /** Sum finite terms with order-independent fixed-point accumulation. */
  template <int Count>
  bool sumTerms(const std::array<double, Count>& terms, double& result) {
    double largest = 0.0;
    for (const double term : terms) {
      if (not std::isfinite(term)) return false;
      largest = std::max(largest, std::abs(term));
    }
    if constexpr (Count == 1) {
      result = terms[0];
      return true;
    } else {
      if (largest > std::nextafter(std::numeric_limits<double>::max() / Count, 0.0)) return false;
      tarch::la::FixedPointSum<Count> accumulator;
      accumulator.clear(largest);
      for (const double term : terms) accumulator.add(term);
      result = accumulator.round();
      return std::isfinite(result);
    }
  }

  /** Cache normal Taylor coefficients for repeated tangential face sampling. */
  template <int N, int V, typename C, typename P>
  class DGFaceExpansion {
    static_assert(N > 0 and V > 0);
    static constexpr int Nodes = peano4::power(N, DIMENSIONS);
    static constexpr int FaceNodes = Nodes / N;
    std::array<double, Nodes * V> _coefficients{};
    std::array<double, N> _positions{};
    int _normal = -1;

    /** Build left-face Taylor weights when generated tables are unavailable. */
    bool makeWeights(std::array<double, N * N>& weights) const {
      using namespace native_neighbour_detail;
      for (int node = 0; node < N; node++) {
        std::array<double, N> polynomial{};
        polynomial[0] = 1.0;
        int degree = 0;
        for (int other = 0; other < N; other++) {
          if (other == node) continue;
          double inverse = 0.0, constant = 0.0;
          if (not ratio(1.0, _positions[node] - _positions[other], inverse)
              or not product(-_positions[other], inverse, constant)) return false;
          std::array<double, N> next{};
          for (int order = 0; order <= degree + 1; order++) {
            double left = 0.0, right = 0.0;
            if (order <= degree and not product(polynomial[order], constant, left)) return false;
            if (order > 0 and not product(polynomial[order - 1], inverse, right)) return false;
            if (not sum(left, right, next[order])) return false;
          }
          polynomial = next;
          degree++;
        }
        for (int order = 0; order < N; order++) weights[order * N + node] = polynomial[order];
      }
      return true;
    }

  public:
    /** Prepare normal Taylor coefficients scaled by normalRatio^order. */
    bool prepare(const P* Q, const double* nodes, int normal, int side, double normalRatio) {
      using namespace native_neighbour_detail;
      _normal = -1;
      if (Q == nullptr or nodes == nullptr or normal < 0 or normal >= DIMENSIONS
          or (side != 0 and side != 1) or not std::isfinite(normalRatio) or normalRatio <= 0.0) return false;
      for (int node = 0; node < N; node++) {
        _positions[node] = node < N / 2 ? 1.0 - nodes[N - 1 - node] : nodes[node];
        if (not std::isfinite(_positions[node]) or _positions[node] < 0.0 or _positions[node] > 1.0
            or (node > 0 and _positions[node - 1] >= _positions[node])) return false;
      }
      std::array<double, N * N> weights{};
      if constexpr (requires { C::taylorWeight(0, 0, 0); }) {
        for (int order = 0; order < N; order++) {
          for (int node = 0; node < N; node++) weights[order * N + node] = C::taylorWeight(side, order, node);
        }
      } else {
        if (not makeWeights(weights)) return false;
        if (side == 1) {
          for (int order = 0; order < N; order++) {
            for (int node = 0; node < N / 2; node++) std::swap(weights[order * N + node], weights[order * N + N - 1 - node]);
            if (order % 2 != 0) for (int node = 0; node < N; node++) weights[order * N + node] = -weights[order * N + node];
          }
        }
      }
      double factor = 1.0;
      for (int order = 0; order < N; order++) {
        if (order > 0 and not product(factor, normalRatio, factor)) return false;
        for (int node = 0; node < N; node++) {
          if (not product(weights[order * N + node], factor, weights[order * N + node])) return false;
        }
      }
      int stride = 1;
      for (int axis = 0; axis < normal; axis++) stride *= N;
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        const int first = (fibre / stride) * (N * stride) + fibre % stride;
        for (int component = 0; component < V; component++) {
          double lower = static_cast<double>(Q[first * V + component]), upper = lower;
          for (int node = 0; node < N; node++) {
            const double value = static_cast<double>(Q[(first + node * stride) * V + component]);
            if (not std::isfinite(value)) return false;
            lower = std::min(lower, value);
            upper = std::max(upper, value);
          }
          const double centre = 0.5 * lower + 0.5 * upper;
          for (int order = 0; order < N; order++) {
            std::array<double, N> terms{};
            for (int node = 0; node < N; node++) {
              double residual = 0.0;
              if (not sum(static_cast<double>(Q[(first + node * stride) * V + component]), -centre, residual)
                  or not product(weights[order * N + node], residual, terms[node])) return false;
            }
            double coefficient = 0.0;
            if (not sumTerms<N>(terms, coefficient)
                or (order == 0 and not sum(centre, coefficient, coefficient))) return false;
            _coefficients[(first + order * stride) * V + component] = coefficient;
          }
        }
      }
      _normal = normal;
      return true;
    }

    /** Evaluate the face Taylor coefficients; ignore the point's normal coordinate. */
    bool sample(const std::array<double, DIMENSIONS>& point, std::array<double, N * V>& jet) const {
      if (_normal < 0) return false;
      std::array<double, DIMENSIONS> centred{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == _normal) continue;
        if (not std::isfinite(point[axis]) or point[axis] < 0.0 or point[axis] > 1.0) return false;
        centred[axis] = point[axis] - 0.5;
      }
      return sampleCentered(centred, jet);
    }

    /** Use centred face coordinates to retain reflection symmetry. */
    bool sampleCentered(const std::array<double, DIMENSIONS>& point, std::array<double, N * V>& jet) const {
      using namespace native_neighbour_detail;
      if (_normal < 0) return false;
      std::array<std::array<double, N>, DIMENSIONS> basis{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == _normal) continue;
        if (not std::isfinite(point[axis]) or point[axis] < -0.5 or point[axis] > 0.5) return false;
        const bool reflected = point[axis] > 0.0;
        const double coordinate = 0.5 - std::abs(point[axis]);
        for (int node = 0; node < N; node++) {
          const int index = reflected ? N - 1 - node : node;
          double value = 1.0;
          for (int other = 0; other < N; other++) {
            if (other == index) continue;
            double factor = 0.0;
            if (not ratio(coordinate - _positions[other], _positions[index] - _positions[other], factor)
                or not product(value, factor, value)) return false;
          }
          basis[axis][node] = value;
        }
      }
      int stride = 1;
      for (int axis = 0; axis < _normal; axis++) stride *= N;
      std::array<double, FaceNodes> weights{};
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        const int first = (fibre / stride) * (N * stride) + fibre % stride;
        int index = first;
        std::array<double, DIMENSIONS - 1> factors{};
        int count = 0;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          const int node = index % N;
          index /= N;
          if (axis != _normal) factors[count++] = basis[axis][node];
        }
        std::sort(factors.begin(), factors.end());
        weights[fibre] = 1.0;
        for (const double factor : factors) if (not product(weights[fibre], factor, weights[fibre])) return false;
      }
      for (int order = 0; order < N; order++) {
        for (int component = 0; component < V; component++) {
          double lower = _coefficients[order * stride * V + component], upper = lower;
          for (int fibre = 1; fibre < FaceNodes; fibre++) {
            const int first = (fibre / stride) * (N * stride) + fibre % stride;
            const double value = _coefficients[(first + order * stride) * V + component];
            lower = std::min(lower, value);
            upper = std::max(upper, value);
          }
          const double centre = 0.5 * lower + 0.5 * upper;
          std::array<double, FaceNodes> terms{};
          for (int fibre = 0; fibre < FaceNodes; fibre++) {
            const int first = (fibre / stride) * (N * stride) + fibre % stride;
            double residual = 0.0;
            if (not sum(_coefficients[(first + order * stride) * V + component], -centre, residual)
                or not product(weights[fibre], residual, terms[fibre])) return false;
          }
          double value = 0.0;
          if (not sumTerms<FaceNodes>(terms, value) or not sum(centre, value, jet[order * V + component])) return false;
        }
      }
      return true;
    }
  };
} // namespace exahype2::amr::resolution_detail
