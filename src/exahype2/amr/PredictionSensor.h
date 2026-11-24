// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "ResolutionSensor.h"

#include <bit>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

// The score of prediction mode. For the face between two cells, the solution is sampled along lines that
// cross the face at right angles. One polynomial is fitted to the values of both cells on such a line. What
// the polynomial cannot reproduce, averaged over the face, is the score of the face: a solution that both
// cells resolve leaves almost nothing. An FV cell also has a score of its own.
namespace exahype2::amr::prediction_detail {
  TARGET_DECLARE
  /** The Gauss-Legendre quadrature rule with Count points on [0,1]. The sensor integrates with it. */
  template <int Count>
  struct GaussRule {
    std::array<double, Count> nodes{}, weights{};
    GPU_CALLABLE_METHOD GaussRule() {
      static_assert(Count > 0);
      const double pi = std::acos(-1.0);
      for (int root = 0; root < (Count + 1) / 2; root++) {
        double x = std::cos(pi * (root + 0.75) / (Count + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < std::numeric_limits<double>::digits; iteration++) {
          double previous = 1.0, value = x;
          if constexpr (Count == 1) value = x;
          for (int degree = 2; degree <= Count; degree++) {
            const double next = ((2 * degree - 1) * x * value - (degree - 1) * previous) / degree;
            previous = value;
            value = next;
          }
          derivative = Count * (x * value - previous) / (x * x - 1.0);
          const double next = x - value / derivative;
          if (next == x) break;
          x = next;
        }
        double previous = 1.0, value = x;
        for (int degree = 2; degree <= Count; degree++) {
          const double next = ((2 * degree - 1) * x * value - (degree - 1) * previous) / degree;
          previous = value;
          value = next;
        }
        derivative = Count * (x * value - previous) / (x * x - 1.0);
        const double weight = 1.0 / ((1.0 - x * x) * derivative * derivative);
        nodes[root] = 0.5 * (1.0 - x);
        nodes[Count - 1 - root] = 1.0 - nodes[root];
        weights[root] = weights[Count - 1 - root] = weight;
      }
    }
  };

  TARGET_DECLARE_END

  /** Returns the rule with Count points. It is computed once. */
  template <int Count>
  const GaussRule<Count>& gaussRule() {
    static const GaussRule<Count> rule;
    return rule;
  }

  TARGET_DECLARE
  /** Returns the Legendre polynomials of the degrees 0 to N-1 at a point of [-1,1]. */
  template <int N>
  GPU_CALLABLE_METHOD std::array<double, N> legendre(double point) {
    std::array<double, N> result{};
    result[0] = 1.0;
    if constexpr (N > 1) result[1] = point;
    for (int degree = 2; degree < N; degree++) {
      result[degree] = ((2 * degree - 1) * point * result[degree - 1]
        - (degree - 1) * result[degree - 2]) / degree;
    }
    return result;
  }

  /** Returns the argument of smaller magnitude if both have the same sign, and zero if not. */
  inline GPU_CALLABLE_METHOD double minmod(double left, double right) {
    if (left == 0.0 or right == 0.0 or std::signbit(left) != std::signbit(right)) return 0.0;
    return std::copysign(std::min(std::abs(left), std::abs(right)), left);
  }

  /** Returns (a-b) divided by the scale of the value. A value that Config leaves out of the score gives zero. */
  template <typename C>
  GPU_CALLABLE_METHOD double resolutionDifference(double a, double b, int component) {
    if constexpr (requires { C::resolutionComponent(component); })
      if (not amr_detail::resolutionComponent<C>(component)) return 0.0;
    return native_neighbour_detail::scaledDifference(a, b, C::stateScale(component));
  }

  /**
   * Returns the limited slope of one value in one volume of an FV cell along one axis, as the change across
   * the volume divided by the scale. An inner volume takes the minmod of the differences to its two
   * neighbours. A volume at the edge of the cell takes the minmod of the two nearest differences. With two
   * volumes per axis the slope is their difference, with one volume it is zero.
   */
  template <int V, typename C, typename P>
  GPU_CALLABLE_METHOD double fvSlope(const AcceptedCellView<P>& view, int volume, int axis, int component) {
    const int count = view.pointsPerAxis;
    if (count == 1) return 0.0;
    const int stride = peano4::power(count, axis);
    const int position = (volume / stride) % count;
    const auto difference = [&](int upper, int lower) {
      return resolutionDifference<C>(
        static_cast<double>(view.values[upper * V + component]),
        static_cast<double>(view.values[lower * V + component]), component);
    };
    if (count == 2) return position == 0 ? difference(volume + stride, volume) : difference(volume, volume - stride);
    if (position == 0) return minmod(difference(volume + stride, volume), difference(volume + 2 * stride, volume + stride));
    if (position == count - 1) return minmod(difference(volume, volume - stride), difference(volume - stride, volume - 2 * stride));
    return minmod(difference(volume, volume - stride), difference(volume + stride, volume));
  }

  /** The slopes of an FV cell and its values relative to a reference state. Each one is computed when it is
   * first asked for. */
  template <int V, typename C, typename P>
  struct CachedFVData {
    struct Entry {
      std::array<double, DIMENSIONS> slopes{};
      double mean = 0.0;
      unsigned int knownSlopes = 0;
      bool knownMean = false;
    };
    AcceptedCellView<P> view;
    const std::array<double, V>& reference;
    Entry* entries;

    GPU_CALLABLE_METHOD double slope(int volume, int axis, int component) const {
      auto& entry = entries[static_cast<std::size_t>(volume) * V + component];
      const unsigned int bit = 1u << axis;
      if ((entry.knownSlopes & bit) == 0) {
        entry.slopes[axis] = fvSlope<V, C>(view, volume, axis, component);
        entry.knownSlopes |= bit;
      }
      return entry.slopes[axis];
    }

    GPU_CALLABLE_METHOD double mean(int volume, int component) const {
      auto& entry = entries[static_cast<std::size_t>(volume) * V + component];
      if (not entry.knownMean) {
        entry.mean = resolutionDifference<C>(
          static_cast<double>(view.values[volume * V + component]),
          reference[component], component);
        entry.knownMean = true;
      }
      return entry.mean;
    }
  };

  TARGET_DECLARE_END

  /** Holds the memory of CachedFVData for one FV cell on the CPU. */
  template <int V, typename C, typename P>
  class FVDataCache {
    using View = CachedFVData<V, C, P>;
    AcceptedCellView<P> _view;
    std::array<double, V> _reference;
    std::vector<typename View::Entry> _entries;

  public:
    explicit FVDataCache(AcceptedCellView<P> view, const std::array<double, V>& reference = {}) :
      _view(view), _reference(reference),
      _entries(view.isFiniteVolume() ? static_cast<std::size_t>(view.numberOfStates()) * V : 0) {}

    View getView() { return {_view, _reference, _entries.data()}; }
  };

  /** Returns the score of one cell without its neighbours. For an FV cell it estimates what the averages do
   * not show: the slope inside the volumes and the mismatch of the linear reconstructions of neighbouring
   * volumes. A DG cell scores zero here. Its score comes from its faces.
   */
  template <int N, int V, typename C, typename P>
  double localDetail(AcceptedCellView<P> view, const double*) {
    if (not view.isFiniteVolume()) {
      int entries = 0;
      return view.pointsPerAxis == N
        and resolution_detail::finitePatch<V, C>(view.values, view.pointsPerAxis, entries)
        ? 0.0 : std::numeric_limits<double>::infinity();
    }
    int volumes = 0;
    if (not resolution_detail::finitePatch<V, C>(view.values, view.pointsPerAxis, volumes))
      return std::numeric_limits<double>::infinity();
    FVDataCache<V, C, P> cache(view);
    const auto data = cache.getView();
    double result = 0.0;
    std::vector<double> representationTerms;
    representationTerms.reserve(static_cast<std::size_t>(volumes) * V * DIMENSIONS);
    for (int volume = 0; volume < volumes; volume++) for (int axis = 0; axis < DIMENSIONS; axis++)
      for (int component = 0; component < V; component++) {
        const double slope = data.slope(volume, axis, component);
        if (not std::isfinite(slope)) return std::numeric_limits<double>::infinity();
        representationTerms.push_back(std::abs(slope) / std::sqrt(12.0 * volumes));
      }
    // Sorting makes the norm independent of the order of the axes and values.
    std::sort(representationTerms.begin(), representationTerms.end());
    for (const double term : representationTerms) result = amr_detail::guardedHypot(result, term);
    int stride = 1;
    for (int normal = 0; normal < DIMENSIONS; normal++, stride *= view.pointsPerAxis) {
      for (int lower = 0; lower < volumes; lower++) {
        if ((lower / stride) % view.pointsPerAxis + 1 == view.pointsPerAxis) continue;
        const int upper = lower + stride;
        std::array<double, V> magnitudes{};
        for (int component = 0; component < V; component++) {
          double norm = 0.0;
          const double mean = resolutionDifference<C>(
            static_cast<double>(view.values[lower * V + component]),
            static_cast<double>(view.values[upper * V + component]), component);
          const double lowerNormal = data.slope(lower, normal, component);
          const double upperNormal = data.slope(upper, normal, component);
          double trace = 0.0;
          if (not native_neighbour_detail::sum(mean, 0.5 * lowerNormal, trace)
              or not native_neighbour_detail::sum(trace, 0.5 * upperNormal, trace))
            return std::numeric_limits<double>::infinity();
          norm = amr_detail::guardedHypot(norm, trace);
          for (int axis = 0; axis < DIMENSIONS; axis++) {
            double delta = 0.0;
            if (not native_neighbour_detail::sum(0.5 * data.slope(lower, axis, component),
                -0.5 * data.slope(upper, axis, component), delta))
              return std::numeric_limits<double>::infinity();
            norm = amr_detail::guardedHypot(norm, delta / std::sqrt(3.0));
          }
          magnitudes[component] = norm;
        }
        std::sort(magnitudes.begin(), magnitudes.end());
        double norm = 0.0;
        for (const double magnitude : magnitudes) norm = amr_detail::guardedHypot(norm, magnitude);
        if (not std::isfinite(norm)) return std::numeric_limits<double>::infinity();
        result = std::max(result, norm);
      }
    }
    return result;
  }

  TARGET_DECLARE
  /** Computes the QR factorisation of the matrix of a least-squares fit, in place. tau receives the
   * Householder factors. Returns false if the matrix does not have full rank. */
  GPU_CALLABLE_METHOD inline bool factorNormalFit(int rows, int columns, double* qr, double* tau) {
    for (int column = 0; column < columns; column++) {
      double norm = 0.0;
      for (int row = column; row < rows; row++) norm = std::hypot(norm, qr[row * columns + column]);
      if (not std::isfinite(norm) or norm == 0.0) return false;
      const double initial = qr[column * columns + column];
      const double diagonal = -std::copysign(norm, initial);
      const double denominator = initial - diagonal;
      tau[column] = (diagonal - initial) / diagonal;
      qr[column * columns + column] = diagonal;
      for (int row = column + 1; row < rows; row++) qr[row * columns + column] /= denominator;
      for (int next = column + 1; next < columns; next++) {
        double dot = qr[column * columns + next];
        for (int row = column + 1; row < rows; row++) dot += qr[row * columns + column] * qr[row * columns + next];
        dot *= tau[column];
        qr[column * columns + next] -= dot;
        for (int row = column + 1; row < rows; row++) qr[row * columns + next] -= qr[row * columns + column] * dot;
      }
    }
    return true;
  }

  /** Returns what a least-squares fit leaves of the samples: the weighted norm of the part that the polynomial
   * cannot reproduce. values holds the samples and is overwritten. */
  template <int V>
  GPU_CALLABLE_METHOD double normalFitResidual(
    int rows, int columns, const double* qr, const double* tau, const double* rootWeights, std::span<double> values
  ) {
    if (values.size() != static_cast<std::size_t>(rows * V)) return std::numeric_limits<double>::infinity();
    std::array<double, V> scales{};
    for (int row = 0; row < rows; row++) for (int component = 0; component < V; component++) {
      if (not std::isfinite(values[row * V + component])) return std::numeric_limits<double>::infinity();
      scales[component] = std::max(scales[component], std::abs(values[row * V + component]));
    }
    for (int row = 0; row < rows; row++) for (int component = 0; component < V; component++) {
      values[row * V + component] = scales[component] == 0.0 ? 0.0
        : (values[row * V + component] / scales[component]) * rootWeights[row];
      if (not std::isfinite(values[row * V + component])) return std::numeric_limits<double>::infinity();
    }
    for (int column = 0; column < columns; column++) for (int component = 0; component < V; component++) {
      double dot = values[column * V + component];
      for (int row = column + 1; row < rows; row++) dot += qr[row * columns + column] * values[row * V + component];
      dot *= tau[column];
      values[column * V + component] -= dot;
      for (int row = column + 1; row < rows; row++) values[row * V + component] -= qr[row * columns + column] * dot;
    }
    std::array<double, V> magnitudes{};
    for (int component = 0; component < V; component++) {
      double norm = 0.0, scaled = 0.0;
      for (int row = columns; row < rows; row++) norm = std::hypot(norm, values[row * V + component]);
      if (not native_neighbour_detail::product(norm, scales[component], scaled)) return std::numeric_limits<double>::infinity();
      magnitudes[component] = scaled;
    }
    COMPILE_IF_ON_HOST(std::sort(magnitudes.begin(), magnitudes.end());)
    COMPILE_IF_ON_DEVICE(detail::transfer_bound::sort(magnitudes.begin(), magnitudes.end());)
    double result = 0.0;
    for (const double magnitude : magnitudes) result = amr_detail::guardedHypot(result, magnitude);
    return result;
  }
  TARGET_DECLARE_END

  /** The least-squares fit of one polynomial to the samples of two cells along a line that crosses their shared
   * face at right angles. A sample is a DG node value or an FV average. The polynomial has as many
   * coefficients as the cell with the fewest samples, at most N. */
  template <int N>
  struct NormalFit {
    int rows = 0, columns = 0;
    bool valid = false;
    std::vector<double> qr, tau, rootWeights;

    NormalFit(int leftCount, int rightCount, bool leftFV, bool rightFV,
              double leftWidth, double rightWidth, const std::array<double, N>& nodes) {
      rows = leftCount + rightCount;
      columns = std::min({N, leftCount, rightCount});
      qr.resize(rows * columns);
      tau.resize(columns);
      rootWeights.resize(rows);
      if (columns <= 0 or rows <= columns) return;
      std::array<double, N> nodalWeights{};
      if (not leftFV or not rightFV) {
        const auto& rule = gaussRule<N>();
        for (int node = 0; node < N; node++) {
          if (not std::isfinite(nodes[node]) or nodes[node] < 0.0 or nodes[node] > 1.0
              or (node > 0 and nodes[node - 1] >= nodes[node])) return;
          for (int quadrature = 0; quadrature < N; quadrature++) {
            double basis = 1.0;
            for (int other = 0; other < N; other++) {
              if (other != node) {
                double factor = 0.0;
                if (not native_neighbour_detail::ratio(rule.nodes[quadrature] - nodes[other], nodes[node] - nodes[other], factor)
                    or not native_neighbour_detail::product(basis, factor, basis)) return;
              }
            }
            double term = 0.0;
            if (not native_neighbour_detail::product(rule.weights[quadrature], basis, term)
                or not native_neighbour_detail::sum(nodalWeights[node], term, nodalWeights[node])) return;
          }
          if (not std::isfinite(nodalWeights[node]) or nodalWeights[node] <= 0.0) return;
        }
      }
      const double width = leftWidth + rightWidth;
      const auto& rule = gaussRule<N>();
      for (int side = 0, first = 0; side < 2; first += side == 0 ? leftCount : rightCount, side++) {
        const int count = side == 0 ? leftCount : rightCount;
        const bool fv = side == 0 ? leftFV : rightFV;
        const double cellWidth = side == 0 ? leftWidth : rightWidth;
        const double offset = side == 0 ? 0.0 : leftWidth;
        for (int node = 0; node < count; node++) {
          const int row = first + node;
          rootWeights[row] = std::sqrt(cellWidth / width * (fv ? 1.0 / count : nodalWeights[node]));
          std::array<double, N> values{};
          if (fv) {
            // An FV sample represents the average of the polynomial over a volume.
            for (int quadrature = 0; quadrature < N; quadrature++) {
              const double position = offset + cellWidth * (node + rule.nodes[quadrature]) / count;
              const auto basis = legendre<N>(2.0 * position / width - 1.0);
              for (int column = 0; column < columns; column++) values[column] += rule.weights[quadrature] * basis[column];
            }
          } else {
            values = legendre<N>(2.0 * (offset + cellWidth * nodes[node]) / width - 1.0);
          }
          for (int column = 0; column < columns; column++) qr[row * columns + column] = rootWeights[row] * values[column];
        }
      }
      valid = factorNormalFit(rows, columns, qr.data(), tau.data());
    }

    /** Returns the part of the samples that the fitted polynomial cannot reproduce. */
    template <int V>
    double residual(std::span<double> values) const {
      if (not valid) return std::numeric_limits<double>::infinity();
      return normalFitResidual<V>(rows, columns, qr.data(), tau.data(), rootWeights.data(), values);
    }
  };

  /** Returns the fit for a pair of cells: their numbers of samples per axis, whether they are FV cells, and
   * their levels. A thread computes each fit once and keeps it. */
  template <int N>
  const NormalFit<N>& normalFit(int leftCount, int rightCount, bool leftFV, bool rightFV,
                               int leftLevel, int rightLevel, const double* originalNodes) {
    std::array<double, N> nodes{};
    if (not leftFV or not rightFV) {
      for (int node = 0; node < N; node++) nodes[node] = originalNodes == nullptr
        ? std::numeric_limits<double>::quiet_NaN()
        : originalNodes[node];
    }
    using Key = std::tuple<int, int, bool, bool, int, std::array<double, N>>;
    // Nodes that are not finite cannot be a key of the cache.
    static thread_local const NormalFit<N> invalid(0, 0, true, true, 1.0, 1.0, {});
    for (double node : nodes) if (not std::isfinite(node)) return invalid;
    static thread_local std::map<Key, NormalFit<N>> cache;
    const Key key{leftCount, rightCount, leftFV, rightFV, leftLevel - rightLevel, nodes};
    const auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    double leftWidth = 1.0, rightWidth = 1.0;
    for (int level = leftLevel; level < rightLevel; level++) leftWidth *= peano4::SubdivisionFactor;
    for (int level = rightLevel; level < leftLevel; level++) rightWidth *= peano4::SubdivisionFactor;
    return cache.try_emplace(key, leftCount, rightCount, leftFV, rightFV, leftWidth, rightWidth, nodes).first->second;
  }

  TARGET_DECLARE
  /** The weights that interpolate a DG cell from its nodes to a point of a face. They give the polynomial on
   * the line through that point at right angles to the face. */
  template <int N>
  struct DGProfileGeometry {
    std::array<double, peano4::power(N, DIMENSIONS - 1)> weights{};
    /** The tangential node that the point lies on, or -1 if the profile needs interpolation. */
    int single{-1};
    bool valid{false};
    GPU_CALLABLE_METHOD DGProfileGeometry(const double* originalNodes, int normal, const std::array<double, DIMENSIONS>& point);
  };

  template <int N>
  GPU_CALLABLE_METHOD DGProfileGeometry<N>::DGProfileGeometry(const double* originalNodes, int normal,
                                         const std::array<double, DIMENSIONS>& point) {
    constexpr int FaceNodes = peano4::power(N, DIMENSIONS - 1);
    // Between two cells of the same level the sample points are the nodes, up to the rounding of the face mapping.
    constexpr double NodeDistance = 16.0 * std::numeric_limits<double>::epsilon();
    const int stride = peano4::power(N, normal);
    std::array<std::array<double, N>, DIMENSIONS> basis{};
    bool onNodes = true;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (axis == normal) continue;
      int nearest = -1;
      for (int node = 0; node < N; node++) {
        if (std::abs(point[axis] - originalNodes[node]) <= NodeDistance) nearest = node;
      }
      if (nearest >= 0) {
        // Use the node itself: its Lagrange weight is exactly one and the others vanish.
        for (int node = 0; node < N; node++) basis[axis][node] = node == nearest ? 1.0 : 0.0;
        continue;
      }
      onNodes = false;
      for (int node = 0; node < N; node++) {
        const double coordinate = originalNodes[node];
        double weight = 1.0;
        for (int other = 0; other < N; other++) {
          const double otherCoordinate = originalNodes[other];
          if (node != other) {
            double factor = 0.0;
            if (not native_neighbour_detail::ratio(point[axis] - otherCoordinate, coordinate - otherCoordinate, factor)
                or not native_neighbour_detail::product(weight, factor, weight)) return;
          }
        }
        basis[axis][node] = weight;
      }
    }
    for (int fibre = 0; fibre < FaceNodes; fibre++) {
      int entry = (fibre / stride) * (N * stride) + fibre % stride;
      double weight = 1.0;
      for (int axis = 0; axis < DIMENSIONS; axis++, entry /= N) {
        if (axis != normal and not native_neighbour_detail::product(weight, basis[axis][entry % N], weight)) return;
      }
      weights[fibre] = weight;
      if (onNodes and weight != 0.0) single = fibre;
    }
    valid = true;
  }

  TARGET_DECLARE_END

  /** Are the weights of DGProfileGeometry cached? Config can switch the cache off. */
  template <typename C>
  constexpr bool cacheProfileGeometry() {
    if constexpr (requires { C::CacheProfileGeometry; }) return C::CacheProfileGeometry;
    return true;
  }

  /**
   * Returns the weights of DGProfileGeometry for a point from the cache of the thread and computes them if
   * the cache does not hold them. The key is the bit pattern of the nodes and of the point, without the
   * coordinate along the normal. The cache has a fixed size. Weights that are not valid are returned in local
   * and not cached.
   */
  template <int N>
  const DGProfileGeometry<N>* cachedDGProfileGeometry(
    const double* originalNodes, int normal, const std::array<double, DIMENSIONS>& point,
    std::optional<DGProfileGeometry<N>>& local
  ) {
    std::array<std::uint64_t, N> nodes{};
    std::array<std::uint64_t, DIMENSIONS - 1> positions{};
    for (int node = 0; node < N; node++) {
      if (not std::isfinite(originalNodes[node])) return nullptr;
      nodes[node] = std::bit_cast<std::uint64_t>(originalNodes[node]);
    }
    for (int axis = 0, tangent = 0; axis < DIMENSIONS; axis++) if (axis != normal) {
      if (not std::isfinite(point[axis])) return nullptr;
      positions[tangent++] = std::bit_cast<std::uint64_t>(point[axis]);
    }
    // A small hash table. The same few sample points are asked for again and again.
    struct Entry {
      int normal{-1};
      std::array<std::uint64_t, N> nodes{};
      std::array<std::uint64_t, DIMENSIONS - 1> positions{};
      std::optional<DGProfileGeometry<N>> geometry;
    };
    constexpr std::size_t Capacity = 4096, Probes = 2;
    static thread_local std::vector<Entry> cache(Capacity);
    std::uint64_t hash = 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(normal + 1);
    for (const auto position : positions) hash = (hash ^ position ^ (position >> 29)) * 0x9E3779B97F4A7C15ULL;
    const std::size_t first = static_cast<std::size_t>(hash >> 52);
    for (std::size_t probe = 0; probe < Probes; probe++) {
      const auto& entry = cache[(first + probe) % Capacity];
      if (entry.geometry and entry.normal == normal and entry.positions == positions and entry.nodes == nodes)
        return &*entry.geometry;
    }
    DGProfileGeometry<N> geometry(originalNodes, normal, point);
    if (not geometry.valid) {
      local.emplace(std::move(geometry));
      return &*local;
    }
    // Take a free place. If there is none, the new entry replaces the first one.
    std::size_t target = first;
    for (std::size_t probe = 0; probe < Probes; probe++) {
      if (not cache[(first + probe) % Capacity].geometry) { target = (first + probe) % Capacity; break; }
    }
    auto& entry = cache[target];
    entry.normal = normal;
    entry.nodes = nodes;
    entry.positions = positions;
    entry.geometry.emplace(std::move(geometry));
    return &*entry.geometry;
  }

  /** Samples the DG polynomial along the line through a face point at right angles to the face. Defined below. */
  template <int N, int V, typename C, typename P>
  bool dgProfile(AcceptedCellView<P> view, const double* originalNodes, int normal,
                 const std::array<double, DIMENSIONS>& point, const std::array<double, V>& reference,
                 std::span<double> values, int first);

  /** Does the work of dgProfile() for the many sample points of one face. The values of the cell relative to
   * the reference state are computed once and used for every point. */
  template <int N, int V, typename C, typename P>
  class DGSourceProfileCache {
    static constexpr int FaceNodes = peano4::power(N, DIMENSIONS - 1);
    struct Group {
      // The values of one layer of nodes and one value, relative to the reference state. They are filled at
      // the first use, which the flags record.
      std::array<double, FaceNodes> data;
      double largest;
      int normalizedCount;
      bool known, valid, constant;
      Group() : largest(0.0), normalizedCount(0), known(false), valid(false), constant(false) {}
    };
    // The cache takes at most 32 KiB. A cell that needs more is interpolated without it.
    static constexpr bool Enabled = sizeof(Group) * static_cast<std::size_t>(N) * V <= 32 * 1024;
    AcceptedCellView<P> _view;
    const std::array<double, V>& _reference;
    int _normal, _stride;
    std::conditional_t<Enabled, std::array<Group, N * V>, std::array<char, 0>> _groups;

    bool prepare(int node, int component, Group& group) {
      if (group.known) return group.valid;
      group.known = true;
      for (int fibre = 0; fibre < FaceNodes; ++fibre) {
        const int entry = (fibre / _stride) * (N * _stride) + fibre % _stride + node * _stride;
        group.data[fibre] = resolutionDifference<C>(
          static_cast<double>(_view.values[entry * V + component]), _reference[component], component);
        if (not std::isfinite(group.data[fibre])) return false;
        group.largest = std::max(group.largest, std::abs(group.data[fibre]));
      }
      group.constant = std::all_of(group.data.begin(), group.data.end(),
        [&](double value) { return value == group.data[0]; });
      group.valid = true;
      return true;
    }

    /** Divide the entries up to last by the largest magnitude, each entry once. */
    static void normalise(Group& group, int last) {
      for (; group.normalizedCount <= last; ++group.normalizedCount) {
        auto& value = group.data[group.normalizedCount];
        value = group.largest == 0.0 ? 0.0 : value / group.largest;
      }
    }

  public:
    DGSourceProfileCache(AcceptedCellView<P> view, int normal, const std::array<double, V>& reference) :
      _view(view), _reference(reference), _normal(normal), _stride(peano4::power(N, normal)) {}

    /** Does what dgProfile() does, with the stored values. */
    bool profile(const double* originalNodes, const std::array<double, DIMENSIONS>& point,
                 std::span<double> values, int first) {
      if constexpr (not Enabled) {
        return dgProfile<N, V, C>(_view, originalNodes, _normal, point, _reference, values, first);
      } else {
        const DGProfileGeometry<N>* geometry = nullptr;
        std::optional<DGProfileGeometry<N>> local;
        if constexpr (cacheProfileGeometry<C>()) geometry = cachedDGProfileGeometry<N>(originalNodes, _normal, point, local);
        if (geometry == nullptr) {
          local.emplace(originalNodes, _normal, point);
          geometry = &*local;
        }
        // Without valid weights there is nothing to interpolate.
        if (not geometry->valid) return false;
        for (int node = 0; node < N; ++node) for (int component = 0; component < V; ++component) {
          auto& group = _groups[node * V + component];
          if (not prepare(node, component, group)) return false;
          if (group.constant) {
            values[(first + node) * V + component] = group.data[0];
            continue;
          }
          if (geometry->single >= 0) {
            // The point is a node, so the sum has one term. A zero is stored as positive zero, as a sum gives it.
            normalise(group, geometry->single);
            const double value = group.data[geometry->single];
            if (not native_neighbour_detail::product(value == 0.0 ? 0.0 : value, group.largest,
                values[(first + node) * V + component])) return false;
            continue;
          }
          normalise(group, FaceNodes - 1);
          std::array<double, FaceNodes> terms{};
          for (int fibre = 0; fibre < FaceNodes; ++fibre) {
            if (not native_neighbour_detail::product(geometry->weights[fibre], group.data[fibre], terms[fibre])) return false;
          }
          double sum = 0.0;
          if (not resolution_detail::sumTerms<FaceNodes>(terms, sum)
              or not native_neighbour_detail::product(sum, group.largest, values[(first + node) * V + component])) return false;
        }
        return true;
      }
    }
  };

  /** Samples the DG polynomial of a cell along the line through a face point at right angles to the face, at
   * the N node positions along that line. values receives the samples from the index first on, relative to
   * the reference state and divided by the scales. */
  template <int N, int V, typename C, typename P>
  bool dgProfile(AcceptedCellView<P> view, const double* originalNodes, int normal,
                 const std::array<double, DIMENSIONS>& point, const std::array<double, V>& reference,
                 std::span<double> values, int first) {
    constexpr int FaceNodes = peano4::power(N, DIMENSIONS - 1);
    const int stride = peano4::power(N, normal);
    const DGProfileGeometry<N>* geometry = nullptr;
    std::optional<DGProfileGeometry<N>> local;
    if constexpr (cacheProfileGeometry<C>()) geometry = cachedDGProfileGeometry<N>(originalNodes, normal, point, local);
    // Use local geometry when caching is disabled or the inputs cannot be cached.
    if (geometry == nullptr) {
      local.emplace(originalNodes, normal, point);
      geometry = &*local;
    }
    if (not geometry->valid) return false;
    const auto& weights = geometry->weights;
    for (int node = 0; node < N; node++) for (int component = 0; component < V; component++) {
      std::array<double, FaceNodes> differences{}, terms{};
      double largest = 0.0;
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        const int entry = (fibre / stride) * (N * stride) + fibre % stride + node * stride;
        differences[fibre] = resolutionDifference<C>(
          static_cast<double>(view.values[entry * V + component]), reference[component], component);
        if (not std::isfinite(differences[fibre])) return false;
        largest = std::max(largest, std::abs(differences[fibre]));
      }
      if (std::all_of(differences.begin(), differences.end(), [&](double value) { return value == differences[0]; })) {
        values[(first + node) * V + component] = differences[0];
        continue;
      }
      if (geometry->single >= 0) {
        // The point is a node, so the sum has one term. A zero is stored as positive zero, as a sum gives it.
        const double value = largest == 0.0 ? 0.0 : differences[geometry->single] / largest;
        if (not native_neighbour_detail::product(value == 0.0 ? 0.0 : value, largest,
            values[(first + node) * V + component])) return false;
        continue;
      }
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        if (not native_neighbour_detail::product(weights[fibre], largest == 0.0 ? 0.0 : differences[fibre] / largest,
            terms[fibre])) return false;
      }
      double sum = 0.0;
      if (not resolution_detail::sumTerms<FaceNodes>(terms, sum)
          or not native_neighbour_detail::product(sum, largest, values[(first + node) * V + component])) return false;
    }
    return true;
  }

  TARGET_DECLARE
  /** Samples an FV cell along the line through a face point at right angles to the face: one value per volume
   * on the line, moved to the point with the slopes across the line. */
  template <int V, typename FVData>
  GPU_CALLABLE_METHOD bool reconstructFVProfile(
    int count, int normal, const std::array<double, DIMENSIONS>& point,
    const FVData& data, std::span<double> values, int first
  ) {
    int tangential = 0, stride = 1;
    std::array<int, DIMENSIONS> index{};
    for (int axis = 0; axis < DIMENSIONS; axis++, stride *= count) {
      if (axis == normal) continue;
      index[axis] = std::clamp(static_cast<int>(std::floor(point[axis] * count)), 0, count - 1);
      tangential += index[axis] * stride;
    }
    const int normalStride = peano4::power(count, normal);
    for (int node = 0; node < count; node++) for (int component = 0; component < V; component++) {
      const int volume = tangential + node * normalStride;
      double value = data.mean(volume, component);
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis != normal) {
          double correction = 0.0;
          if (not native_neighbour_detail::product(data.slope(volume, axis, component),
              point[axis] * count - (index[axis] + 0.5), correction)
              or not native_neighbour_detail::sum(value, correction, value)) return false;
        }
      }
      if (not std::isfinite(value)) return false;
      values[(first + node) * V + component] = value;
    }
    return true;
  }

  TARGET_DECLARE_END

  /** reconstructFVProfile() for an FV cell whose slopes are cached. */
  template <int V, typename C, typename P>
  bool fvProfile(AcceptedCellView<P> view, int normal, const std::array<double, DIMENSIONS>& point,
                 FVDataCache<V, C, P>& cache, std::span<double> values, int first) {
    return reconstructFVProfile<V>(view.pointsPerAxis, normal, point, cache.getView(), values, first);
  }

  /** reconstructFVProfile() for an FV cell without a cache. */
  template <int V, typename C, typename P>
  bool fvProfile(AcceptedCellView<P> view, int normal, const std::array<double, DIMENSIONS>& point,
                 const std::array<double, V>& reference, std::span<double> values, int first) {
    FVDataCache<V, C, P> data(view, reference);
    return fvProfile<V, C>(view, normal, point, data, values, first);
  }

  /** Returns the score of the face between two cells in prediction mode. At sample points of the face it takes
   * the values of both cells along the line that crosses the face at right angles, and fits one polynomial to
   * them. The score is the root mean square over the face of what the fits leave.
   */
  template <int N, int V, typename C, typename P>
  double faceDetail(AcceptedCellView<P> own, AcceptedCellView<P> other,
                    const NativeCellKey& ownKey, const NativeCellKey& otherKey,
                    const double* nodes, int normal, int ownSide) {
    static_assert(N > 0 and V > 0);
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    if (normal < 0 or normal >= DIMENSIONS or (ownSide != 0 and ownSide != 1)
        or (not own.isFiniteVolume() and own.pointsPerAxis != N)
        or (not other.isFiniteVolume() and other.pointsPerAxis != N)
        or not resolution_detail::finitePatch<V, C>(own.values, own.pointsPerAxis, count)
        or not resolution_detail::finitePatch<V, C>(other.values, other.pointsPerAxis, count)) return invalid;
    native_neighbour_detail::FaceProjection ownMap, otherMap;
    if (not native_neighbour_detail::projectLatticeFace(ownKey, otherKey, normal, ownMap, otherMap)) return invalid;
    const bool ownIsLeft = ownSide == 1;
    const auto left = ownIsLeft ? own : other;
    const auto right = ownIsLeft ? other : own;
    const auto& leftKey = ownIsLeft ? ownKey : otherKey;
    const auto& rightKey = ownIsLeft ? otherKey : ownKey;
    const auto& leftMap = ownIsLeft ? ownMap : otherMap;
    const auto& rightMap = ownIsLeft ? otherMap : ownMap;
    const auto& fit = normalFit<N>(left.pointsPerAxis, right.pointsPerAxis, left.isFiniteVolume(),
      right.isFiniteVolume(), leftKey[0], rightKey[0], nodes);
    if (not fit.valid) return invalid;
    std::array<double, V> reference{};
    for (int component = 0; component < V; component++) reference[component] = static_cast<double>(left.values[component]);
    constexpr int TangentialOrder = N > 1 ? N : 2;
    const auto& rule = gaussRule<TangentialOrder>();
    if (not left.isFiniteVolume() and not right.isFiniteVolume()) {
      std::array<double, 2 * N * V> values{};
      DGSourceProfileCache<N, V, C, P> leftSource(left, normal, reference), rightSource(right, normal, reference);
      constexpr std::size_t Samples = peano4::power(TangentialOrder, DIMENSIONS - 1);
      double result = 0.0;
      for (std::size_t sample = 0; sample < Samples; sample++) {
        std::size_t index = sample;
        double weight = 1.0;
        std::array<double, DIMENSIONS> leftPoint{}, rightPoint{};
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          if (axis == normal) continue;
          const std::size_t position = index % TangentialOrder;
          index /= TangentialOrder;
          const double width = 0.5 - (-0.5);
          const double point = -0.5 + width * rule.nodes[position];
          leftPoint[axis] = 0.5 + leftMap.offset[axis] + point * leftMap.ratio[axis];
          rightPoint[axis] = 0.5 + rightMap.offset[axis] + point * rightMap.ratio[axis];
          weight *= width * rule.weights[position];
        }
        if (not leftSource.profile(nodes, leftPoint, values, 0)
            or not rightSource.profile(nodes, rightPoint, values, left.pointsPerAxis)) return invalid;
        const double residual = fit.template residual<V>(values);
        result = amr_detail::guardedHypot(result, std::sqrt(weight) * residual);
        if (not std::isfinite(result)) return invalid;
      }
      return result;
    }
    FVDataCache<V, C, P> leftData(left, reference), rightData(right, reference);
    const auto& fineKey = ownKey[0] >= otherKey[0] ? ownKey : otherKey;
    std::array<std::vector<double>, DIMENSIONS> positions, weights;
    std::size_t samples = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (axis == normal) { positions[axis].push_back(0.0); weights[axis].push_back(1.0); continue; }
      std::vector<double> boundaries{-0.5, 0.5};
      if (own.isFiniteVolume()) resolution_detail::addFVBoundaries(boundaries, own.pointsPerAxis, ownKey, fineKey, axis);
      if (other.isFiniteVolume()) resolution_detail::addFVBoundaries(boundaries, other.pointsPerAxis, otherKey, fineKey, axis);
      std::sort(boundaries.begin(), boundaries.end());
      boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
      for (std::size_t interval = 1; interval < boundaries.size(); interval++) {
        const double width = boundaries[interval] - boundaries[interval - 1];
        for (int quadrature = 0; quadrature < TangentialOrder; quadrature++) {
          positions[axis].push_back(boundaries[interval - 1] + width * rule.nodes[quadrature]);
          weights[axis].push_back(width * rule.weights[quadrature]);
        }
      }
      if (samples > std::numeric_limits<std::size_t>::max() / positions[axis].size()) return invalid;
      samples *= positions[axis].size();
    }
    std::vector<double> values(static_cast<std::size_t>(fit.rows) * V);
    std::optional<DGSourceProfileCache<N, V, C, P>> leftSource, rightSource;
    if (not left.isFiniteVolume()) leftSource.emplace(left, normal, reference);
    if (not right.isFiniteVolume()) rightSource.emplace(right, normal, reference);
    double result = 0.0;
    for (std::size_t sample = 0; sample < samples; sample++) {
      std::size_t index = sample;
      double weight = 1.0;
      std::array<double, DIMENSIONS> leftPoint{}, rightPoint{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == normal) continue;
        const std::size_t position = index % positions[axis].size();
        index /= positions[axis].size();
        const double point = positions[axis][position];
        leftPoint[axis] = 0.5 + leftMap.offset[axis] + point * leftMap.ratio[axis];
        rightPoint[axis] = 0.5 + rightMap.offset[axis] + point * rightMap.ratio[axis];
        weight *= weights[axis][position];
      }
      const bool leftValid = left.isFiniteVolume()
        ? fvProfile<V, C>(left, normal, leftPoint, leftData, values, 0)
        : leftSource->profile(nodes, leftPoint, values, 0);
      if (not leftValid) return invalid;
      const bool rightValid = right.isFiniteVolume()
        ? fvProfile<V, C>(right, normal, rightPoint, rightData, values, left.pointsPerAxis)
        : rightSource->profile(nodes, rightPoint, values, left.pointsPerAxis);
      if (not rightValid) return invalid;
      const double residual = fit.template residual<V>(values);
      result = amr_detail::guardedHypot(result, std::sqrt(weight) * residual);
      if (not std::isfinite(result)) return invalid;
    }
    return result;
  }
} // namespace exahype2::amr::prediction_detail

TARGET_DECLARE
namespace exahype2::amr::prediction_detail {
  /** NormalFit for a GPU kernel, with arrays of fixed size. MaxPoints is the largest number of samples of a cell. */
  template <int N, int MaxPoints>
  struct DeviceNormalFit {
    int rows = 0, columns = 0;
    bool valid = false;
    std::array<double, 2 * MaxPoints * N> qr{};
    std::array<double, N> tau{};
    std::array<double, 2 * MaxPoints> rootWeights{};

    GPU_CALLABLE_METHOD DeviceNormalFit(int leftCount, int rightCount, bool leftFV, bool rightFV,
              double leftWidth, double rightWidth, const std::array<double, N>& nodes) {
      static_assert(MaxPoints >= N);
      if (leftCount <= 0 or leftCount > MaxPoints or rightCount <= 0 or rightCount > MaxPoints) return;
      rows = leftCount + rightCount;
      columns = std::min(N, std::min(leftCount, rightCount));
      if (columns <= 0 or rows <= columns) return;
      const GaussRule<N> rule;
      std::array<double, N> nodalWeights{};
      if (not leftFV or not rightFV) {
        for (int node = 0; node < N; node++) {
          if (not std::isfinite(nodes[node]) or nodes[node] < 0.0 or nodes[node] > 1.0
              or (node > 0 and nodes[node - 1] >= nodes[node])) return;
          for (int quadrature = 0; quadrature < N; quadrature++) {
            double basis = 1.0;
            for (int other = 0; other < N; other++) {
              if (other != node) {
                double factor = 0.0;
                if (not native_neighbour_detail::ratio(rule.nodes[quadrature] - nodes[other], nodes[node] - nodes[other], factor)
                    or not native_neighbour_detail::product(basis, factor, basis)) return;
              }
            }
            double term = 0.0;
            if (not native_neighbour_detail::product(rule.weights[quadrature], basis, term)
                or not native_neighbour_detail::sum(nodalWeights[node], term, nodalWeights[node])) return;
          }
          if (not std::isfinite(nodalWeights[node]) or nodalWeights[node] <= 0.0) return;
        }
      }
      const double width = leftWidth + rightWidth;
      for (int side = 0, first = 0; side < 2; first += side == 0 ? leftCount : rightCount, side++) {
        const int count = side == 0 ? leftCount : rightCount;
        const bool fv = side == 0 ? leftFV : rightFV;
        const double cellWidth = side == 0 ? leftWidth : rightWidth;
        const double offset = side == 0 ? 0.0 : leftWidth;
        for (int node = 0; node < count; node++) {
          const int row = first + node;
          rootWeights[row] = std::sqrt(cellWidth / width * (fv ? 1.0 / count : nodalWeights[node]));
          std::array<double, N> values{};
          if (fv) {
            // An FV sample represents the average of the polynomial over a volume.
            for (int quadrature = 0; quadrature < N; quadrature++) {
              const double position = offset + cellWidth * (node + rule.nodes[quadrature]) / count;
              const auto basis = legendre<N>(2.0 * position / width - 1.0);
              for (int column = 0; column < columns; column++) values[column] += rule.weights[quadrature] * basis[column];
            }
          } else {
            values = legendre<N>(2.0 * (offset + cellWidth * nodes[node]) / width - 1.0);
          }
          for (int column = 0; column < columns; column++) qr[row * columns + column] = rootWeights[row] * values[column];
        }
      }
      valid = factorNormalFit(rows, columns, qr.data(), tau.data());
    }

    template <int V>
    GPU_CALLABLE_METHOD double residual(std::span<double> values) const {
      if (not valid) return std::numeric_limits<double>::infinity();
      return normalFitResidual<V>(rows, columns, qr.data(), tau.data(), rootWeights.data(), values);
    }
  };

  /** dgProfile() for a GPU kernel. */
  template <int N, int V, typename C, typename P>
  GPU_CALLABLE_METHOD bool deviceDGProfile(AcceptedCellView<P> view, const double* originalNodes, int normal,
                 const std::array<double, DIMENSIONS>& point, const std::array<double, V>& reference,
                 std::span<double> values, int first) {
    constexpr int FaceNodes = peano4::power(N, DIMENSIONS - 1);
    const int stride = peano4::power(N, normal);
    if (originalNodes == nullptr) return false;
    const DGProfileGeometry<N> geometry(originalNodes, normal, point);
    if (not geometry.valid) return false;
    const auto& weights = geometry.weights;
    for (int node = 0; node < N; node++) for (int component = 0; component < V; component++) {
      std::array<double, FaceNodes> differences{}, terms{};
      double largest = 0.0;
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        const int entry = (fibre / stride) * (N * stride) + fibre % stride + node * stride;
        differences[fibre] = resolutionDifference<C>(
          static_cast<double>(view.values[entry * V + component]), reference[component], component);
        if (not std::isfinite(differences[fibre])) return false;
        largest = std::max(largest, std::abs(differences[fibre]));
      }
      bool constant = true;
      for (int fibre = 1; fibre < FaceNodes; ++fibre) constant = constant and differences[fibre] == differences[0];
      if (constant) {
        values[(first + node) * V + component] = differences[0];
        continue;
      }
      for (int fibre = 0; fibre < FaceNodes; fibre++) {
        if (not native_neighbour_detail::product(weights[fibre], largest == 0.0 ? 0.0 : differences[fibre] / largest,
            terms[fibre])) return false;
      }
      double sum = 0.0;
      if (not resolution_detail::sumTerms<FaceNodes>(terms, sum)
          or not native_neighbour_detail::product(sum, largest, values[(first + node) * V + component])) return false;
    }
    return true;
  }

  /** The slopes of an FV cell and its values relative to a reference state, for a GPU kernel. Nothing is
   * stored: every call computes its result. */
  template <int V, typename C, typename P>
  struct DeviceFVData {
    AcceptedCellView<P> view;
    std::array<double, V> reference{};
    GPU_CALLABLE_METHOD explicit DeviceFVData(AcceptedCellView<P> cell, const std::array<double, V>& origin = {}) :
      view(cell), reference(origin) {}
    GPU_CALLABLE_METHOD double slope(int volume, int axis, int component) const {
      return fvSlope<V, C>(view, volume, axis, component);
    }
    GPU_CALLABLE_METHOD double mean(int volume, int component) const {
      return resolutionDifference<C>(static_cast<double>(view.values[volume * V + component]), reference[component], component);
    }
  };

  /** The score of localDetail() for a GPU kernel. MaxPoints is the largest number of nodes or volumes per
   * axis that a cell may have.
   */
  template <int N, int V, typename C, int MaxPoints, typename P>
  GPU_CALLABLE_METHOD double deviceLocalDetail(AcceptedCellView<P> view, const double*) {
    static_assert(MaxPoints >= N);
    if (view.pointsPerAxis > MaxPoints) return std::numeric_limits<double>::infinity();
    if (not view.isFiniteVolume()) {
      int entries = 0;
      return view.pointsPerAxis == N
        and resolution_detail::finitePatch<V, C>(view.values, view.pointsPerAxis, entries)
        ? 0.0 : std::numeric_limits<double>::infinity();
    }
    int volumes = 0;
    if (not resolution_detail::finitePatch<V, C>(view.values, view.pointsPerAxis, volumes))
      return std::numeric_limits<double>::infinity();
    DeviceFVData<V, C, P> data(view);
    double result = 0.0;
    detail::transfer_bound::FixedVector<double, peano4::power(MaxPoints, DIMENSIONS) * V * DIMENSIONS> representationTerms;
    for (int volume = 0; volume < volumes; volume++) for (int axis = 0; axis < DIMENSIONS; axis++)
      for (int component = 0; component < V; component++) {
        const double slope = data.slope(volume, axis, component);
        if (not std::isfinite(slope)) return std::numeric_limits<double>::infinity();
        if (not representationTerms.push_back(std::abs(slope) / std::sqrt(12.0 * volumes)))
          return std::numeric_limits<double>::infinity();
      }
    // Sorting makes the norm independent of the order of the axes and values.
    detail::transfer_bound::sort(representationTerms.begin(), representationTerms.end());
    for (const double term : representationTerms) result = amr_detail::guardedHypot(result, term);
    int stride = 1;
    for (int normal = 0; normal < DIMENSIONS; normal++, stride *= view.pointsPerAxis) {
      for (int lower = 0; lower < volumes; lower++) {
        if ((lower / stride) % view.pointsPerAxis + 1 == view.pointsPerAxis) continue;
        const int upper = lower + stride;
        std::array<double, V> magnitudes{};
        for (int component = 0; component < V; component++) {
          double norm = 0.0;
          const double mean = resolutionDifference<C>(
            static_cast<double>(view.values[lower * V + component]),
            static_cast<double>(view.values[upper * V + component]), component);
          const double lowerNormal = data.slope(lower, normal, component);
          const double upperNormal = data.slope(upper, normal, component);
          double trace = 0.0;
          if (not native_neighbour_detail::sum(mean, 0.5 * lowerNormal, trace)
              or not native_neighbour_detail::sum(trace, 0.5 * upperNormal, trace))
            return std::numeric_limits<double>::infinity();
          norm = amr_detail::guardedHypot(norm, trace);
          for (int axis = 0; axis < DIMENSIONS; axis++) {
            double delta = 0.0;
            if (not native_neighbour_detail::sum(0.5 * data.slope(lower, axis, component),
                -0.5 * data.slope(upper, axis, component), delta))
              return std::numeric_limits<double>::infinity();
            norm = amr_detail::guardedHypot(norm, delta / std::sqrt(3.0));
          }
          magnitudes[component] = norm;
        }
        detail::transfer_bound::sort(magnitudes.begin(), magnitudes.end());
        double norm = 0.0;
        for (const double magnitude : magnitudes) norm = amr_detail::guardedHypot(norm, magnitude);
        if (not std::isfinite(norm)) return std::numeric_limits<double>::infinity();
        result = std::max(result, norm);
      }
    }
    return result;
  }

  /** The score of faceDetail() for a GPU kernel. MaxPoints is the largest number of nodes or volumes per axis
   * that a cell may have.
   */
  template <int N, int V, typename C, int MaxPoints, typename P>
  GPU_CALLABLE_METHOD double deviceFaceDetail(AcceptedCellView<P> own, AcceptedCellView<P> other,
                    const NativeCellKey& ownKey, const NativeCellKey& otherKey,
                    const double* nodes, int normal, int ownSide) {
    static_assert(N > 0 and V > 0 and MaxPoints >= N);
    const double invalid = std::numeric_limits<double>::infinity();
    int count = 0;
    if (own.pointsPerAxis > MaxPoints or other.pointsPerAxis > MaxPoints or normal < 0 or normal >= DIMENSIONS or (ownSide != 0 and ownSide != 1)
        or (not own.isFiniteVolume() and own.pointsPerAxis != N)
        or (not other.isFiniteVolume() and other.pointsPerAxis != N)
        or not resolution_detail::finitePatch<V, C>(own.values, own.pointsPerAxis, count)
        or not resolution_detail::finitePatch<V, C>(other.values, other.pointsPerAxis, count)) return invalid;
    native_neighbour_detail::FaceProjection ownMap, otherMap;
    if (not native_neighbour_detail::projectLatticeFace(ownKey, otherKey, normal, ownMap, otherMap)) return invalid;
    const bool ownIsLeft = ownSide == 1;
    const auto left = ownIsLeft ? own : other;
    const auto right = ownIsLeft ? other : own;
    const auto& leftKey = ownIsLeft ? ownKey : otherKey;
    const auto& rightKey = ownIsLeft ? otherKey : ownKey;
    const auto& leftMap = ownIsLeft ? ownMap : otherMap;
    const auto& rightMap = ownIsLeft ? otherMap : ownMap;
    std::array<double, N> originalNodes{};
    if (not left.isFiniteVolume() or not right.isFiniteVolume()) {
      if (nodes == nullptr) return invalid;
      for (int node = 0; node < N; ++node) {
        if (not std::isfinite(nodes[node])) return invalid;
        originalNodes[node] = nodes[node];
      }
    }
    double leftWidth = 1.0, rightWidth = 1.0;
    for (int level = leftKey[0]; level < rightKey[0]; ++level) leftWidth *= peano4::SubdivisionFactor;
    for (int level = rightKey[0]; level < leftKey[0]; ++level) rightWidth *= peano4::SubdivisionFactor;
    const DeviceNormalFit<N, MaxPoints> fit(left.pointsPerAxis, right.pointsPerAxis, left.isFiniteVolume(),
      right.isFiniteVolume(), leftWidth, rightWidth, originalNodes);
    if (not fit.valid) return invalid;
    std::array<double, V> reference{};
    for (int component = 0; component < V; component++) reference[component] = static_cast<double>(left.values[component]);
    constexpr int TangentialOrder = N > 1 ? N : 2;
    const auto rule = GaussRule<TangentialOrder>{};
    if (not left.isFiniteVolume() and not right.isFiniteVolume()) {
      std::array<double, 2 * N * V> values{};
      constexpr std::size_t Samples = peano4::power(TangentialOrder, DIMENSIONS - 1);
      double result = 0.0;
      for (std::size_t sample = 0; sample < Samples; sample++) {
        std::size_t index = sample;
        double weight = 1.0;
        std::array<double, DIMENSIONS> leftPoint{}, rightPoint{};
        for (int axis = 0; axis < DIMENSIONS; axis++) {
          if (axis == normal) continue;
          const std::size_t position = index % TangentialOrder;
          index /= TangentialOrder;
          const double width = 0.5 - (-0.5);
          const double point = -0.5 + width * rule.nodes[position];
          leftPoint[axis] = 0.5 + leftMap.offset[axis] + point * leftMap.ratio[axis];
          rightPoint[axis] = 0.5 + rightMap.offset[axis] + point * rightMap.ratio[axis];
          weight *= width * rule.weights[position];
        }
        if (not deviceDGProfile<N, V, C>(left, nodes, normal, leftPoint, reference, values, 0)
            or not deviceDGProfile<N, V, C>(right, nodes, normal, rightPoint, reference, values, left.pointsPerAxis)) return invalid;
        const double residual = fit.template residual<V>(values);
        result = amr_detail::guardedHypot(result, std::sqrt(weight) * residual);
        if (not std::isfinite(result)) return invalid;
      }
      return result;
    }
    DeviceFVData<V, C, P> leftData(left, reference), rightData(right, reference);
    const auto& fineKey = ownKey[0] >= otherKey[0] ? ownKey : otherKey;
    constexpr int MaxSamplesPerAxis = (2 * MaxPoints + 1) * TangentialOrder;
    std::array<detail::transfer_bound::FixedVector<double, MaxSamplesPerAxis>, DIMENSIONS> positions, weights;
    std::size_t samples = 1;
    for (int axis = 0; axis < DIMENSIONS; axis++) {
      if (axis == normal) {
        if (not positions[axis].push_back(0.0) or not weights[axis].push_back(1.0)) return invalid;
        continue;
      }
      detail::transfer_bound::FixedVector<double, 2 * MaxPoints + 2> boundaries;
      if (not boundaries.push_back(-0.5) or not boundaries.push_back(0.5)) return invalid;
      if (own.isFiniteVolume() and not resolution_detail::deviceAddFVBoundaries(boundaries, own.pointsPerAxis, ownKey, fineKey, axis)) return invalid;
      if (other.isFiniteVolume() and not resolution_detail::deviceAddFVBoundaries(boundaries, other.pointsPerAxis, otherKey, fineKey, axis)) return invalid;
      detail::transfer_bound::sort(boundaries.begin(), boundaries.end());
      boundaries.resize(static_cast<int>(detail::transfer_bound::unique(boundaries.begin(), boundaries.end()) - boundaries.begin()));
      for (int interval = 1; interval < boundaries.size(); interval++) {
        const double width = boundaries[interval] - boundaries[interval - 1];
        for (int quadrature = 0; quadrature < TangentialOrder; quadrature++) {
          if (not positions[axis].push_back(boundaries[interval - 1] + width * rule.nodes[quadrature])
              or not weights[axis].push_back(width * rule.weights[quadrature])) return invalid;
        }
      }
      if (samples > std::numeric_limits<std::size_t>::max() / positions[axis].size()) return invalid;
      samples *= positions[axis].size();
    }
    std::array<double, 2 * MaxPoints * V> valueStorage{};
    const std::span<double> values(valueStorage.data(), static_cast<std::size_t>(fit.rows) * V);
    double result = 0.0;
    for (std::size_t sample = 0; sample < samples; sample++) {
      std::size_t index = sample;
      double weight = 1.0;
      std::array<double, DIMENSIONS> leftPoint{}, rightPoint{};
      for (int axis = 0; axis < DIMENSIONS; axis++) {
        if (axis == normal) continue;
        const std::size_t position = index % positions[axis].size();
        index /= positions[axis].size();
        const double point = positions[axis][position];
        leftPoint[axis] = 0.5 + leftMap.offset[axis] + point * leftMap.ratio[axis];
        rightPoint[axis] = 0.5 + rightMap.offset[axis] + point * rightMap.ratio[axis];
        weight *= weights[axis][position];
      }
      const bool leftValid = left.isFiniteVolume()
        ? reconstructFVProfile<V>(left.pointsPerAxis, normal, leftPoint, leftData, values, 0)
        : deviceDGProfile<N, V, C>(left, nodes, normal, leftPoint, reference, values, 0);
      if (not leftValid) return invalid;
      const bool rightValid = right.isFiniteVolume()
        ? reconstructFVProfile<V>(right.pointsPerAxis, normal, rightPoint, rightData, values, left.pointsPerAxis)
        : deviceDGProfile<N, V, C>(right, nodes, normal, rightPoint, reference, values, left.pointsPerAxis);
      if (not rightValid) return invalid;
      const double residual = fit.template residual<V>(values);
      result = amr_detail::guardedHypot(result, std::sqrt(weight) * residual);
      if (not std::isfinite(result)) return invalid;
    }
    return result;
  }
} // namespace exahype2::amr::prediction_detail
TARGET_DECLARE_END
