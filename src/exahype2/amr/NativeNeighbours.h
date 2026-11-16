// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "peano4/CellConnectivity.h"
#include "tarch/la/FixedPointSum.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace exahype2::amr {
  /** Cell level and domain-relative lattice indices. */
  using NativeCellKey = std::array<int, DIMENSIONS + 1>;

  namespace native_neighbour_detail {
    struct FaceProjection {
      std::array<double, DIMENSIONS> offset{}, ratio{};
    };

    inline bool product(double left, double right, double& result) {
      const double largest = std::numeric_limits<double>::max();
      if (not std::isfinite(left) or not std::isfinite(right)) return false;
      if (std::abs(right) > 1.0) {
        // Prevent speculative division from overflowing.
        volatile double divisor = std::abs(right);
        if (std::abs(left) > std::nextafter(largest / divisor, 0.0)) return false;
      }
      if (std::abs(left) <= 1.e150 and std::abs(right) <= 1.e150) result = left * right;
      else {
        // Prevent speculative multiplication before the overflow check.
        volatile double bounded = left;
        result = bounded * right;
      }
      return std::isfinite(result);
    }

    inline bool sum(double left, double right, double& result) {
      const double largest = std::numeric_limits<double>::max();
      if (not std::isfinite(left) or not std::isfinite(right)
          or (std::signbit(left) == std::signbit(right) and std::abs(left) > largest - std::abs(right))) return false;
      if (std::abs(left) <= largest / 2.0 and std::abs(right) <= largest / 2.0) result = left + right;
      else {
        volatile double bounded = left;
        result = bounded + right;
      }
      return std::isfinite(result);
    }

    inline bool ratio(double numerator, double denominator, double& result) {
      const double largest = std::numeric_limits<double>::max();
      if (not std::isfinite(numerator) or not std::isfinite(denominator) or denominator == 0.0) return false;
      if (std::abs(denominator) < 1.0) {
        volatile double factor = std::abs(denominator);
        if (std::abs(numerator) > std::nextafter(largest * factor, 0.0)) return false;
      }
      if (std::abs(numerator) <= 1.e150 and std::abs(denominator) >= 1.e-150) result = numerator / denominator;
      else {
        volatile double bounded = numerator;
        result = bounded / denominator;
      }
      return std::isfinite(result);
    }

    /** Find leaf cells on the requested side of a face. */
    template <typename Callback>
    bool visitFaceSide(
      const peano4::CellConnectivity& mesh, int face, int side, int remaining,
      Callback& callback
    ) {
      if (face < 0 or face >= mesh.getNumberOfFaces() or remaining < 0) return false;
      const int slot = mesh.getFaceCell(face, side);
      if (slot >= 0) {
        callback(slot);
        return true;
      }
      for (const auto adjacency : {peano4::Adjacency::Local, peano4::Adjacency::Remote}) {
        for (const auto& jump : mesh.getLevelJumps(adjacency)) {
          if (jump.coarseFace == face and side != jump.coarseSide) {
            bool complete = true;
            for (const int fineFace : jump.fineFaces) {
              complete = visitFaceSide(mesh, fineFace, side, remaining - 1, callback) and complete;
            }
            return complete;
          }
          if (side == jump.coarseSide
              and std::find(std::begin(jump.fineFaces), std::end(jump.fineFaces), face) != std::end(jump.fineFaces)) {
            return visitFaceSide(mesh, jump.coarseFace, side, remaining - 1, callback);
          }
        }
      }
      // Report missing or remote leaf data as an incomplete neighbourhood.
      return false;
    }

    inline double scaledDifference(double left, double right, double scale) {
      const double invalid = std::numeric_limits<double>::infinity();
      const double largest = std::numeric_limits<double>::max();
      if (not std::isfinite(left) or not std::isfinite(right)
          or not std::isfinite(scale) or scale <= 0.0) return invalid;
      if (std::signbit(left) != std::signbit(right) and std::abs(left) > largest - std::abs(right)) {
        if (scale <= 1.0) return invalid;
        const double divisor = std::max(1.0, scale);
        left /= divisor;
        right /= divisor;
        double result = 0.0;
        return sum(left, -right, result) ? result : invalid;
      }
      double delta = 0.0, result = 0.0;
      return sum(left, -right, delta) and ratio(delta, scale, result) ? result : invalid;
    }

    /** Evaluate a tensor-product DG polynomial in reference coordinates. */
    template <int N, int V, typename P>
    bool samplePolynomial(const P* Q, const double* nodes, const std::array<double, DIMENSIONS>& point,
                          std::array<double, V>& output) {
      if (Q == nullptr or nodes == nullptr) return false;
      constexpr int Count = peano4::power(N, DIMENSIONS);
      std::array<std::array<double, N>, DIMENSIONS> basis{};
      // Use symmetric node positions on reflected cells.
      std::array<double, N> positions{};
      for (int node = 0; node < N; node++) {
        positions[node] = node < N / 2 ? 1.0 - nodes[N - 1 - node] : nodes[node];
        if (not std::isfinite(positions[node]) or positions[node] < 0.0 or positions[node] > 1.0
            or (node > 0 and positions[node - 1] >= positions[node])) return false;
      }
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        const bool reflected = point[axis] > 0.5;
        const double coordinate = reflected ? 1.0 - point[axis] : point[axis];
        for (int node = 0; node < N; node++) {
          const int index = reflected ? N - 1 - node : node;
          double value = 1.0;
          for (int other = 0; other < N; other++) {
            if (other != index) {
              double factor = 0.0;
              if (not ratio(coordinate - positions[other], positions[index] - positions[other], factor)
                  or not product(value, factor, value)) return false;
            }
          }
          if (not std::isfinite(value)) return false;
          basis[axis][node] = value;
        }
      }
      for (int component = 0; component < V; component++) {
        double lower = static_cast<double>(Q[component]), upper = lower;
        for (int node = 0; node < Count; node++) {
          const double value = static_cast<double>(Q[node * V + component]);
          if (not std::isfinite(value)) return false;
          lower = std::min(lower, value);
          upper = std::max(upper, value);
        }
        if (lower == upper) {
          output[component] = lower;
          continue;
        }
        // Centre nodal values before interpolation to limit cancellation and overflow.
        const double centre = 0.5 * lower + 0.5 * upper;
        std::array<double, Count> terms{};
        double largestTerm = 0.0;
        for (int node = 0; node < Count; node++) {
          std::array<double, DIMENSIONS> factors{};
          int index = node;
          for (int axis = 0; axis < DIMENSIONS; axis++) {
            factors[axis] = basis[axis][index % N];
            index /= N;
          }
          std::sort(factors.begin(), factors.end());
          double weight = 1.0;
          for (const double factor : factors) if (not product(weight, factor, weight)) return false;
          double residual = 0.0, term = 0.0;
          if (not sum(static_cast<double>(Q[node * V + component]), -centre, residual)
              or not product(weight, residual, term)) return false;
          terms[node] = term;
          largestTerm = std::max(largestTerm, std::abs(term));
        }
        // Bound the sum independently of node order before fixed-point rounding.
        if (largestTerm > std::nextafter(std::numeric_limits<double>::max() / Count, 0.0)) return false;
        tarch::la::FixedPointSum<Count> accumulator;
        accumulator.clear(largestTerm);
        for (const double term : terms) accumulator.add(term);
        if (not sum(centre, accumulator.round(), output[component])) return false;
      }
      return true;
    }

    template <int N, int V, typename P>
    bool sampleVolumes(const P* Q, const std::array<double, DIMENSIONS>& point,
                       std::array<double, V>& output) {
      if (Q == nullptr) return false;
      int index = 0, stride = 1;
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        const int volume = std::clamp(static_cast<int>(std::floor(point[axis] * N)), 0, N - 1);
        index += stride * volume;
        stride *= N;
      }
      for (int component = 0; component < V; component++) {
        output[component] = static_cast<double>(Q[index * V + component]);
        if (not std::isfinite(output[component])) return false;
      }
      return true;
    }

    /** Map a physical interval into a cell, allowing for coordinate roundoff. */
    inline bool projectPhysicalInterval(
      double centre, double width, double targetCentre, double targetWidth,
      double& offset, double& fraction
    ) {
      const double scale = std::max({std::abs(centre), std::abs(targetCentre), targetWidth});
      const double uncertainty = 64.0 * std::numeric_limits<double>::epsilon() * scale;
      // Reject intervals too small to locate reliably.
      if (uncertainty >= 0.25 * width) return false;
      double distance = 0.0, room = 0.0;
      if (not sum(centre, -targetCentre, distance) or not sum(targetWidth, -width, room)) return false;
      if (std::abs(distance) > 0.5 * room and std::abs(distance) - 0.5 * room > uncertainty) return false;
      return ratio(distance, targetWidth, offset) and ratio(width, targetWidth, fraction);
    }

    /** Validate the level and indices against the representable lattice extent. */
    inline bool validKey(const NativeCellKey& key) {
      if (key[0] < 0) return false;
      int extent = 1;
      for (int level = 0; level < key[0]; level++) {
        if (extent > std::numeric_limits<int>::max() / peano4::SubdivisionFactor) return false;
        extent *= peano4::SubdivisionFactor;
      }
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (key[axis + 1] < 0 or key[axis + 1] >= extent) return false;
      }
      return true;
    }

    /** Map a finer ternary face into its containing coarse face. */
    inline bool projectLatticeFace(
      const NativeCellKey& own, const NativeCellKey& other, int normal,
      FaceProjection& ownMap, FaceProjection& otherMap
    ) {
      if (not validKey(own) or not validKey(other)) return false;
      const bool ownIsFine = own[0] >= other[0];
      const auto& fine = ownIsFine ? own : other;
      const auto& coarse = ownIsFine ? other : own;
      auto& fineMap = ownIsFine ? ownMap : otherMap;
      auto& coarseMap = ownIsFine ? otherMap : ownMap;
      // validKey bounds factor and the signed numerator by INT_MAX.
      long long factor = 1;
      for (int level = coarse[0]; level < fine[0]; level++) factor *= peano4::SubdivisionFactor;
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == normal) continue;
        if (fine[axis + 1] / factor != coarse[axis + 1]) return false;
        const long long child = fine[axis + 1] % factor;
        fineMap.ratio[axis] = 1.0;
        coarseMap.ratio[axis] = 1.0 / static_cast<double>(factor);
        // Reflecting child to factor-1-child negates the offset exactly.
        coarseMap.offset[axis] = 0.5 * static_cast<double>(child - (factor - 1 - child))
          / static_cast<double>(factor);
      }
      return true;
    }

    /** Return the maximum scaled vector jump over samples on the common subface. */
    template <int N, int V, typename C, typename P>
    double sampleFaceJump(
      const P* own, const P* other, const FaceProjection& ownMap, const FaceProjection& otherMap,
      const double* nodes, int normal, int ownSide
    ) {
      const double invalid = std::numeric_limits<double>::infinity();
      std::array<double, DIMENSIONS> ownPoint{}, otherPoint{};
      ownPoint[normal] = ownSide;
      otherPoint[normal] = 1 - ownSide;
      double result = 0.0;
      constexpr int Samples = peano4::power(N, DIMENSIONS - 1);
      for (int sample = 0; sample < Samples; sample++) {
        int index = sample;
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          if (axis == normal) continue;
          const int point = index % N;
          index /= N;
          const double coordinate = C::IsFiniteVolume ? (point + 0.5) / N : nodes[point];
          ownPoint[axis] = std::clamp(0.5 + ownMap.offset[axis] + (coordinate - 0.5) * ownMap.ratio[axis], 0.0, 1.0);
          otherPoint[axis] = std::clamp(0.5 + otherMap.offset[axis] + (coordinate - 0.5) * otherMap.ratio[axis], 0.0, 1.0);
        }
        std::array<double, V> ownValue{}, otherValue{}, magnitudes{};
        if constexpr (C::IsFiniteVolume) {
          if (not sampleVolumes<N, V>(own, ownPoint, ownValue)
              or not sampleVolumes<N, V>(other, otherPoint, otherValue)) return invalid;
        } else {
          if (not samplePolynomial<N, V>(own, nodes, ownPoint, ownValue)
              or not samplePolynomial<N, V>(other, nodes, otherPoint, otherValue)) return invalid;
        }
        for (int component = 0; component < V; component++) {
          magnitudes[component] = std::abs(scaledDifference(ownValue[component], otherValue[component], C::stateScale(component)));
        }
        std::sort(magnitudes.begin(), magnitudes.end());
        const double largest = magnitudes.back();
        if (not std::isfinite(largest)) return invalid;
        if (largest == 0.0) continue;
        double unitNorm = 0.0;
        for (const double magnitude : magnitudes) unitNorm = std::hypot(unitNorm, magnitude / largest);
        if (unitNorm > 1.0) {
          volatile double divisor = unitNorm;
          if (largest > std::nextafter(std::numeric_limits<double>::max() / divisor, 0.0)) return invalid;
        }
        double norm = 0.0;
        if (not product(largest, unitNorm, norm)) return invalid;
        result = std::max(result, norm);
      }
      return result;
    }
  } // namespace native_neighbour_detail

  /**
   * Visit local leaf neighbours across faces and level jumps.
   * callback(neighbour, normal, ownSide) identifies this cell's lower (0) or upper (1) face.
   * Callbacks may also accept only the neighbour slot.
   * Physical boundaries are skipped; missing interior data returns false.
   */
  template <typename Callback>
  bool forEachNativeNeighbour(const peano4::CellConnectivity& mesh, int slot, Callback callback) {
    if (slot < 0 or slot >= mesh.getNumberOfSlots()) return false;
    bool complete = true;
    for (int faceNumber = 0; faceNumber < 2 * DIMENSIONS; faceNumber++) {
      const int face = mesh.getCellFace(slot, faceNumber);
      if (face < 0 or face >= mesh.getNumberOfFaces()) { complete = false; continue; }
      if (mesh.getFaceKind(face) == peano4::FaceKind::Boundary) continue;
      const int normal = faceNumber % DIMENSIONS;
      const int ownSide = faceNumber < DIMENSIONS ? 0 : 1;
      // ownSide is the neighbour's side in the face-to-cell adjacency.
      const int neighbourSide = ownSide;
      auto visit = [&](int neighbour) {
        if constexpr (std::is_invocable_v<Callback&, int, int, int>) callback(neighbour, normal, ownSide);
        else callback(neighbour);
      };
      complete = native_neighbour_detail::visitFaceSide(
        mesh, face, neighbourSide, mesh.getMaximumLevel() + 2, visit
      ) and complete;
    }
    return complete;
  }

  /**
   * Measure jumps at the finer face's DG nodes or FV volume centres.
   * normal/ownSide identify the face, including periodic faces.
   */
  template <int N, int V, typename C, typename P>
  double nativeFaceJump(
    const P* own, const P* other,
    const tarch::la::Vector<DIMENSIONS, double>& ownCentre,
    const tarch::la::Vector<DIMENSIONS, double>& ownWidth,
    const tarch::la::Vector<DIMENSIONS, double>& otherCentre,
    const tarch::la::Vector<DIMENSIONS, double>& otherWidth,
    const double* nodes, int normal, int ownSide
  ) {
    static_assert(N > 0 and V > 0);
    const double invalid = std::numeric_limits<double>::infinity();
    if (own == nullptr or other == nullptr or normal < 0 or normal >= DIMENSIONS
        or (ownSide != 0 and ownSide != 1)) return invalid;
    if constexpr (not C::IsFiniteVolume) if (nodes == nullptr) return invalid;
    native_neighbour_detail::FaceProjection ownMap, otherMap;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (not std::isfinite(ownCentre(axis)) or not std::isfinite(otherCentre(axis))
          or not std::isfinite(ownWidth(axis)) or ownWidth(axis) <= 0.0
          or not std::isfinite(otherWidth(axis)) or otherWidth(axis) <= 0.0) return invalid;
      if (axis == normal) continue;
      const double width = std::min(ownWidth(axis), otherWidth(axis));
      const double centre = ownWidth(axis) <= otherWidth(axis) ? ownCentre(axis) : otherCentre(axis);
      if (not native_neighbour_detail::projectPhysicalInterval(
            centre, width, ownCentre(axis), ownWidth(axis), ownMap.offset[axis], ownMap.ratio[axis])
          or not native_neighbour_detail::projectPhysicalInterval(
            centre, width, otherCentre(axis), otherWidth(axis), otherMap.offset[axis], otherMap.ratio[axis])) return invalid;
    }
    return native_neighbour_detail::sampleFaceJump<N, V, C>(own, other, ownMap, otherMap, nodes, normal, ownSide);
  }

  /**
   * Use lattice indices to locate the common face without coordinate roundoff.
   * normal/ownSide also handle periodic faces.
   */
  template <int N, int V, typename C, typename P>
  double nativeFaceJump(
    const P* own, const P* other, const NativeCellKey& ownKey, const NativeCellKey& otherKey,
    const double* nodes, int normal, int ownSide
  ) {
    static_assert(N > 0 and V > 0);
    const double invalid = std::numeric_limits<double>::infinity();
    if (own == nullptr or other == nullptr or normal < 0 or normal >= DIMENSIONS
        or (ownSide != 0 and ownSide != 1)) return invalid;
    if constexpr (not C::IsFiniteVolume) if (nodes == nullptr) return invalid;
    native_neighbour_detail::FaceProjection ownMap, otherMap;
    if (not native_neighbour_detail::projectLatticeFace(ownKey, otherKey, normal, ownMap, otherMap)) return invalid;
    return native_neighbour_detail::sampleFaceJump<N, V, C>(own, other, ownMap, otherMap, nodes, normal, ownSide);
  }
} // namespace exahype2::amr
