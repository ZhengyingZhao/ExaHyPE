// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "AMRPolicies.h"
#include "DGDetail.h"

#include <vector>

namespace exahype2::amr::resolution_detail {
  TARGET_DECLARE
  /** Validate the accepted layout, all stored values, and component scales. */
  template <int V, typename C, typename P>
  GPU_CALLABLE_METHOD bool finitePatch(const P* Q, int pointsPerAxis, int& count, bool* constant = nullptr,
                   std::array<double, V>* lower = nullptr, std::array<double, V>* upper = nullptr) {
    if (constant != nullptr) *constant = false;
    if (Q == nullptr or pointsPerAxis <= 0) return false;
    count = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (count > std::numeric_limits<int>::max() / pointsPerAxis) return false;
      count *= pointsPerAxis;
    }
    if (count > std::numeric_limits<int>::max() / V) return false;
    for (int component = 0; component < V; component++) {
      const double scale = C::stateScale(component);
      if (not std::isfinite(scale) or scale <= 0.0) return false;
    }
    if (lower != nullptr) lower->fill(std::numeric_limits<double>::infinity());
    if (upper != nullptr) upper->fill(-std::numeric_limits<double>::infinity());
    bool same = true;
    for (int entry = 0; entry < count * V; entry++) {
      const double value = static_cast<double>(Q[entry]);
      if (not std::isfinite(value)) return false;
      if (constant != nullptr) same = same and Q[entry] == Q[entry % V];
      if (lower != nullptr) (*lower)[entry % V] = std::min((*lower)[entry % V], value);
      if (upper != nullptr) (*upper)[entry % V] = std::max((*upper)[entry % V], value);
    }
    if (constant != nullptr) *constant = same;
    return true;
  }

  /** Component-scaled Euclidean distance, independent of component ordering. */
  template <int V, typename C, typename Left, typename Right>
  GPU_CALLABLE_METHOD double stateDifference(const Left* left, const Right* right) {
    std::array<double, V> magnitudes{};
    for (int component = 0; component < V; component++) {
      magnitudes[component] = std::abs(native_neighbour_detail::scaledDifference(
        static_cast<double>(left[component]), static_cast<double>(right[component]), C::stateScale(component)
      ));
    }
    detail::transfer_bound::sort(magnitudes.begin(), magnitudes.end());
    double norm = 0.0;
    for (const double magnitude : magnitudes) norm = amr_detail::guardedHypot(norm, magnitude);
    return norm;
  }

  /** Return the largest adjacent FV jump, or infinity for invalid or over-budget data. */
  template <int V, typename C, typename P>
  GPU_CALLABLE_METHOD double fvDetail(const P* Q, int pointsPerAxis, double budget = std::numeric_limits<double>::infinity()) {
    static_assert(V > 0);
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    bool constant = false;
    if (std::isnan(budget) or budget < 0.0 or not finitePatch<V, C>(Q, pointsPerAxis, count, &constant)) return invalid;
    if (constant) return 0.0;
    std::array<double, V> scales{};
    for (int component = 0; component < V; component++) scales[component] = C::stateScale(component);
    // Accumulate safe squared magnitudes; use hypot for extreme values.
    const double smallestSquareRoot = std::sqrt(std::numeric_limits<double>::min());
    const double largestSquareRoot = 0.5 * std::sqrt(std::numeric_limits<double>::max() / V);
    double squaredBudget = invalid;
    if (std::isfinite(budget) and budget < 0.5 * std::sqrt(std::numeric_limits<double>::max())) {
      // Keep the squared gate below budget^2; confirm excess after sqrt.
      squaredBudget = budget <= 2.0 * smallestSquareRoot ? 0.0 : std::nextafter(budget * budget, 0.0);
    }
    double largestSquared = 0.0, largestExtreme = 0.0;
    int stride = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++, stride *= pointsPerAxis) {
      for (int volume = 0; volume < count; volume++) {
        if ((volume / stride) % pointsPerAxis + 1 < pointsPerAxis) {
          std::array<double, V> magnitudes{};
          bool squaredIsSafe = true;
          for (int component = 0; component < V; component++) {
            const double magnitude = std::abs(native_neighbour_detail::scaledDifference(
              static_cast<double>(Q[volume * V + component]),
              static_cast<double>(Q[(volume + stride) * V + component]), scales[component]
            ));
            if (not std::isfinite(magnitude)) return invalid;
            magnitudes[component] = magnitude;
            squaredIsSafe = squaredIsSafe and magnitude <= largestSquareRoot
              and (magnitude == 0.0 or magnitude >= smallestSquareRoot);
          }
          detail::transfer_bound::sort(magnitudes.begin(), magnitudes.end());
          if (squaredIsSafe) {
            double squared = 0.0;
            for (const double magnitude : magnitudes) squared += magnitude * magnitude;
            if (squared > squaredBudget and std::sqrt(squared) > budget) return invalid;
            largestSquared = std::max(largestSquared, squared);
          } else {
            double norm = 0.0;
            for (const double magnitude : magnitudes) norm = amr_detail::guardedHypot(norm, magnitude);
            if (norm > budget) return invalid;
            largestExtreme = std::max(largestExtreme, norm);
          }
        }
      }
    }
    const double result = std::max(std::sqrt(largestSquared), largestExtreme);
    return result > budget ? invalid : result;
  }

  /** Test the jump between neighbouring volume averages. When the patch's range fits the budget,
   * the score can be the budget itself.
   */
  template <int V, typename C, typename P>
  GPU_CALLABLE_METHOD double fvResolutionDetail(const P* Q, int pointsPerAxis, double budget) {
    static_assert(V > 0);
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    bool constant = false;
    std::array<double, V> lower{}, upper{}, ranges{};
    if (std::isnan(budget) or budget < 0.0
        or not finitePatch<V, C>(Q, pointsPerAxis, count, &constant, &lower, &upper)) return invalid;
    if (constant) return 0.0;
    // Every adjacent component difference lies within the patch's component range.
    if (amr_detail::transfer_bound::arithmeticSupported()) {
      for (int component = 0; component < V; component++) {
        ranges[component] = amr_detail::transfer_bound::magnitude(amr_detail::transfer_bound::difference(
          upper[component], lower[component], C::stateScale(component)
        ));
      }
      if (amr_detail::transfer_bound::normPasses<V>(ranges, budget)) return budget;
    }
    return fvDetail<V, C>(Q, pointsPerAxis, budget);
  }

  /** Read the FV value at a face point. At a subcell boundary, use the mean of the touching values. */
  template <int V, typename P>
  GPU_CALLABLE_METHOD bool sampleFVFace(const P* Q, int pointsPerAxis, const std::array<double, DIMENSIONS>& point,
                    int normal, int side, std::array<double, V>& value) {
    std::array<int, DIMENSIONS> lower{}, upper{};
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (axis == normal) {
        lower[axis] = upper[axis] = side == 0 ? 0 : pointsPerAxis - 1;
        continue;
      }
      const double position = point[axis] * pointsPerAxis;
      const double boundary = std::round(position);
      const double uncertainty = 32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(position));
      if (boundary > 0.0 and boundary < pointsPerAxis and std::abs(position - boundary) <= uncertainty) {
        lower[axis] = static_cast<int>(boundary) - 1;
        upper[axis] = static_cast<int>(boundary);
      } else {
        lower[axis] = upper[axis] = std::clamp(static_cast<int>(std::floor(position)), 0, pointsPerAxis - 1);
      }
    }
    constexpr int Samples = 1 << (DIMENSIONS - 1);
    for (int component = 0; component < V; component++) {
      double samples[Samples];
      for (int sample = 0; sample < Samples; sample++) {
        int index = 0, stride = 1, bit = 0;
        for (int axis = 0; axis < DIMENSIONS; axis++, stride *= pointsPerAxis) {
          const bool upperSide = axis != normal and ((sample >> bit++) & 1) != 0;
          index += stride * (upperSide ? upper[axis] : lower[axis]);
        }
        samples[sample] = static_cast<double>(Q[index * V + component]);
      }
      bool same = true;
      for (int sample = 1; sample < Samples; sample++) same = same and samples[sample] == samples[0];
      if (same) value[component] = samples[0];
      else {
        // Reserve headroom for the fixed-point accumulator.
        for (const double sample : samples) if (std::abs(sample) > std::numeric_limits<double>::max() / 2.0) return false;
        value[component] = tarch::la::meanInFixedPoint(samples);
      }
      if (not std::isfinite(value[component])) return false;
    }
    return true;
  }

  TARGET_DECLARE_END

  /** Add FV boundaries on the common finer face, using lattice arithmetic. */
  inline void addFVBoundaries(std::vector<double>& boundaries, int pointsPerAxis,
                              const NativeCellKey& key, const NativeCellKey& fineKey, int axis) {
    long long factor = 1;
    for (int level = key[0]; level < fineKey[0]; level++) factor *= peano4::SubdivisionFactor;
    const long long child = fineKey[axis + 1] % factor;
    const long long offset = child * pointsPerAxis;
    const long long first = offset / factor + 1;
    for (long long volume = first; volume < pointsPerAxis; volume++) {
      const long long numerator = volume * factor - offset;
      if (numerator >= pointsPerAxis) break;
      // Centred coordinates make reflected boundaries exact negatives.
      boundaries.push_back(static_cast<double>(2 * numerator - pointsPerAxis) / (2.0 * pointsPerAxis));
    }
  }

  TARGET_DECLARE
  /** Add the subcell boundaries that cross the finer face along this axis. */
  template <typename Container>
  GPU_CALLABLE_METHOD bool deviceAddFVBoundaries(Container& boundaries, int pointsPerAxis,
      const NativeCellKey& key, const NativeCellKey& fineKey, int axis) {
    long long factor = 1;
    for (int level = key[0]; level < fineKey[0]; level++) factor *= peano4::SubdivisionFactor;
    const long long child = fineKey[axis + 1] % factor;
    const long long offset = child * pointsPerAxis;
    const long long first = offset / factor + 1;
    for (long long volume = first; volume < pointsPerAxis; volume++) {
      const long long numerator = volume * factor - offset;
      if (numerator >= pointsPerAxis) break;
      if (not boundaries.push_back(static_cast<double>(2 * numerator - pointsPerAxis) / (2.0 * pointsPerAxis))) return false;
    }
    return true;
  }

  /** Sample one side of the face at a point, using the FV value or the DG Taylor coefficients there. */
  template <int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD bool sampleFace(
    AcceptedCellView<P> view, const DGFaceExpansion<N, V, C, P>& expansion,
    const std::array<double, DIMENSIONS>& point, int normal, int side,
    std::array<double, N * V>& jet
  ) {
    if (not view.isFiniteVolume()) return expansion.sampleCentered(point, jet);
    std::array<double, DIMENSIONS> position{};
    for (int axis = 0; axis < DIMENSIONS; axis++) position[axis] = 0.5 + point[axis];
    std::array<double, V> value{};
    if (not sampleFVFace<V>(view.values, view.pointsPerAxis, position, normal, side, value)) return false;
    for (int component = 0; component < V; component++) jet[component] = value[component];
    return true;
  }
  TARGET_DECLARE_END

  /** Measure the largest scaled jump across a shared face, using the accepted DG or FV data on each side. */
  template <int N, int V, typename C, typename P>
  double faceDetail(AcceptedCellView<P> own, AcceptedCellView<P> other,
                    const NativeCellKey& ownKey, const NativeCellKey& otherKey,
                    const double* nodes, int normal, int ownSide) {
    static_assert(N > 0 and V > 0);
    using native_neighbour_detail::FaceProjection;
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    if (normal < 0 or normal >= DIMENSIONS or (ownSide != 0 and ownSide != 1)
        or (not own.isFiniteVolume() and own.pointsPerAxis != N)
        or (not other.isFiniteVolume() and other.pointsPerAxis != N)
        or not finitePatch<V, C>(own.values, own.pointsPerAxis, count)
        or not finitePatch<V, C>(other.values, other.pointsPerAxis, count)) return invalid;
    FaceProjection ownMap, otherMap;
    if (not native_neighbour_detail::projectLatticeFace(ownKey, otherKey, normal, ownMap, otherMap)) return invalid;
    const auto& fineKey = ownKey[0] >= otherKey[0] ? ownKey : otherKey;
    double ownWidth = 1.0, otherWidth = 1.0;
    for (int level = ownKey[0]; level < fineKey[0]; level++) ownWidth *= peano4::SubdivisionFactor;
    for (int level = otherKey[0]; level < fineKey[0]; level++) otherWidth *= peano4::SubdivisionFactor;
    const double halfWidth = 0.5 * std::min(
      ownWidth / (own.isFiniteVolume() ? own.pointsPerAxis : 1),
      otherWidth / (other.isFiniteVolume() ? other.pointsPerAxis : 1)
    );
    DGFaceExpansion<N, V, C, P> ownExpansion, otherExpansion;
    if ((not own.isFiniteVolume() and not ownExpansion.prepare(own.values, nodes, normal, ownSide, halfWidth / ownWidth))
        or (not other.isFiniteVolume() and not otherExpansion.prepare(other.values, nodes, normal, 1 - ownSide, halfWidth / otherWidth))) return invalid;
    std::array<std::vector<double>, DIMENSIONS> coordinates;
    std::size_t samples = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      auto& points = coordinates[axis];
      if (axis == normal) { points.push_back(0.0); continue; }
      if (own.isFiniteVolume() or other.isFiniteVolume()) {
        std::vector<double> boundaries{-0.5, 0.5};
        if (own.isFiniteVolume()) addFVBoundaries(boundaries, own.pointsPerAxis, ownKey, fineKey, axis);
        if (other.isFiniteVolume()) addFVBoundaries(boundaries, other.pointsPerAxis, otherKey, fineKey, axis);
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (std::size_t i = 1; i < boundaries.size(); i++) points.push_back(0.5 * boundaries[i - 1] + 0.5 * boundaries[i]);
      }
      if (not own.isFiniteVolume() or not other.isFiniteVolume()) {
        if (nodes == nullptr) return invalid;
        for (int node = 0; node < N; node++) {
          const double point = node < N / 2 ? -(nodes[N - 1 - node] - 0.5) : nodes[node] - 0.5;
          if (not std::isfinite(point) or point < -0.5 or point > 0.5) return invalid;
          points.push_back(point);
        }
      }
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());
      if (points.empty() or samples > std::numeric_limits<std::size_t>::max() / points.size()) return invalid;
      samples *= points.size();
    }
    double result = 0.0;
    for (std::size_t sample = 0; sample < samples; sample++) {
      std::size_t index = sample;
      std::array<double, DIMENSIONS> ownPoint{}, otherPoint{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == normal) continue;
        const double point = coordinates[axis][index % coordinates[axis].size()];
        index /= coordinates[axis].size();
        ownPoint[axis] = std::clamp(ownMap.offset[axis] + point * ownMap.ratio[axis], -0.5, 0.5);
        otherPoint[axis] = std::clamp(otherMap.offset[axis] + point * otherMap.ratio[axis], -0.5, 0.5);
      }
      std::array<double, N * V> ownJet{}, otherJet{};
      if (not sampleFace<N, V, C>(own, ownExpansion, ownPoint, normal, ownSide, ownJet)
          or not sampleFace<N, V, C>(other, otherExpansion, otherPoint, normal, 1 - ownSide, otherJet)) return invalid;
      for (int order = 0; order < N; order++) {
        result = std::max(result, stateDifference<V, C>(ownJet.data() + order * V, otherJet.data() + order * V));
      }
    }
    return result;
  }

  TARGET_DECLARE
  /** Measure the largest scaled jump across a shared face, using the accepted DG or FV data on each side. */
  template <int N, int V, typename C, int MaxPoints, typename P>
  GPU_CALLABLE_METHOD double deviceFaceDetail(AcceptedCellView<P> own, AcceptedCellView<P> other,
                    const NativeCellKey& ownKey, const NativeCellKey& otherKey,
                    const double* nodes, int normal, int ownSide) {
    static_assert(N > 0 and V > 0 and MaxPoints >= N);
    using native_neighbour_detail::FaceProjection;
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    if (own.pointsPerAxis > MaxPoints or other.pointsPerAxis > MaxPoints
        or normal < 0 or normal >= DIMENSIONS or (ownSide != 0 and ownSide != 1)
        or (not own.isFiniteVolume() and own.pointsPerAxis != N)
        or (not other.isFiniteVolume() and other.pointsPerAxis != N)
        or not finitePatch<V, C>(own.values, own.pointsPerAxis, count)
        or not finitePatch<V, C>(other.values, other.pointsPerAxis, count)) return invalid;
    FaceProjection ownMap, otherMap;
    if (not native_neighbour_detail::projectLatticeFace(ownKey, otherKey, normal, ownMap, otherMap)) return invalid;
    const auto& fineKey = ownKey[0] >= otherKey[0] ? ownKey : otherKey;
    double ownWidth = 1.0, otherWidth = 1.0;
    for (int level = ownKey[0]; level < fineKey[0]; level++) ownWidth *= peano4::SubdivisionFactor;
    for (int level = otherKey[0]; level < fineKey[0]; level++) otherWidth *= peano4::SubdivisionFactor;
    const double halfWidth = 0.5 * std::min(
      ownWidth / (own.isFiniteVolume() ? own.pointsPerAxis : 1),
      otherWidth / (other.isFiniteVolume() ? other.pointsPerAxis : 1)
    );
    DGFaceExpansion<N, V, C, P> ownExpansion, otherExpansion;
    if ((not own.isFiniteVolume() and not ownExpansion.prepare(own.values, nodes, normal, ownSide, halfWidth / ownWidth))
        or (not other.isFiniteVolume() and not otherExpansion.prepare(other.values, nodes, normal, 1 - ownSide, halfWidth / otherWidth))) return invalid;
    using Coordinates = detail::transfer_bound::FixedVector<double, 2 * MaxPoints + N + 1>;
    std::array<Coordinates, DIMENSIONS> coordinates;
    std::size_t samples = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      auto& points = coordinates[axis];
      if (axis == normal) { points.push_back(0.0); continue; }
      if (own.isFiniteVolume() or other.isFiniteVolume()) {
        detail::transfer_bound::FixedVector<double, 2 * MaxPoints + 2> boundaries;
        boundaries.push_back(-0.5);
        boundaries.push_back(0.5);
        if (own.isFiniteVolume() and not deviceAddFVBoundaries(boundaries, own.pointsPerAxis, ownKey, fineKey, axis)) return invalid;
        if (other.isFiniteVolume() and not deviceAddFVBoundaries(boundaries, other.pointsPerAxis, otherKey, fineKey, axis)) return invalid;
        detail::transfer_bound::sort(boundaries.begin(), boundaries.end());
        boundaries.resize(int(detail::transfer_bound::unique(boundaries.begin(), boundaries.end()) - boundaries.begin()));
        for (int i = 1; i < boundaries.size(); i++) points.push_back(0.5 * boundaries[i - 1] + 0.5 * boundaries[i]);
      }
      if (not own.isFiniteVolume() or not other.isFiniteVolume()) {
        if (nodes == nullptr) return invalid;
        for (int node = 0; node < N; node++) {
          const double point = node < N / 2 ? -(nodes[N - 1 - node] - 0.5) : nodes[node] - 0.5;
          if (not std::isfinite(point) or point < -0.5 or point > 0.5) return invalid;
          points.push_back(point);
        }
      }
      detail::transfer_bound::sort(points.begin(), points.end());
      points.resize(int(detail::transfer_bound::unique(points.begin(), points.end()) - points.begin()));
      if (points.empty() or samples > std::numeric_limits<std::size_t>::max() / points.size()) return invalid;
      samples *= points.size();
    }
    double result = 0.0;
    for (std::size_t sample = 0; sample < samples; sample++) {
      std::size_t index = sample;
      std::array<double, DIMENSIONS> ownPoint{}, otherPoint{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == normal) continue;
        const double point = coordinates[axis][index % coordinates[axis].size()];
        index /= coordinates[axis].size();
        ownPoint[axis] = std::clamp(ownMap.offset[axis] + point * ownMap.ratio[axis], -0.5, 0.5);
        otherPoint[axis] = std::clamp(otherMap.offset[axis] + point * otherMap.ratio[axis], -0.5, 0.5);
      }
      std::array<double, N * V> ownJet{}, otherJet{};
      if (not sampleFace<N, V, C>(own, ownExpansion, ownPoint, normal, ownSide, ownJet)
          or not sampleFace<N, V, C>(other, otherExpansion, otherPoint, normal, 1 - ownSide, otherJet)) return invalid;
      for (int order = 0; order < N; order++) {
        result = std::max(result, stateDifference<V, C>(ownJet.data() + order * V, otherJet.data() + order * V));
      }
    }
    return result;
  }
  TARGET_DECLARE_END
} // namespace exahype2::amr::resolution_detail
