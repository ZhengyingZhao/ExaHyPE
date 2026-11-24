// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "TransferBounds.h"
#include <algorithm>
#include <type_traits>
#include <cstddef>

// When siblings coarsen, their parent keeps one DG polynomial. This file bounds how far the polynomial of a
// child lies from that parent polynomial at the nodes of the child. The bounds are intervals that contain
// the exact difference, rounding included.
namespace exahype2::amr::detail::retained_parent {
  namespace bound = ::exahype2::amr::detail::transfer_bound;
  using Interval = bound::Interval;
  /** Stands in for Geometry and PreparedParent in a criterion for FV, which needs neither. */
  struct NoGeometry {};

  TARGET_DECLARE
  /** Returns bounds of a/b for two intervals. A denominator that contains zero gives no bound. */
  GPU_CALLABLE_METHOD inline Interval quotient(Interval a, Interval b) {
    if (not bound::valid(a) or not bound::valid(b) or (b.lo <= 0. and b.hi >= 0.)) return bound::bad();
    if (bound::zero(a)) return {0., 0.};
    const std::array<double, 2> numerator{a.lo, a.hi}, denominator{b.lo, b.hi};
    std::array<double, 4> values;
    int entry = 0;
    for (double x : numerator) for (double y : denominator) {
      const double divisor = std::abs(y);
      if (divisor < 1.) {
        // x/y overflows if |x| exceeds the largest double times |y|. This limit is rounded towards zero.
        const double safe = std::nextafter(bound::roundedMultiply(std::numeric_limits<double>::max(), divisor), 0.);
        if (std::abs(x) > safe) return bound::bad();
      }
      values[entry++] = bound::roundedDivide(x, y);
    }
    for (double x : values) if (not std::isfinite(x)) return bound::bad();
    double lower = values[0], upper = values[0];
    for (int entry = 1; entry < 4; entry++) {
      lower = std::min(lower, values[entry]);
      upper = std::max(upper, values[entry]);
    }
    return {bound::down(lower), bound::up(upper)};
  }
  TARGET_DECLARE_END

  /** The weights that evaluate the DG polynomial of a parent at the nodes of its children, along one axis.
   * For every child position, entry (row, col) bounds the value of the parent's basis function col at the
   * child's node row.
   */
  template<int N, int R>
  class Geometry {
    static_assert(N > 0 and R > 1);
    std::array<double, N> _nodes{};
    std::array<std::array<Interval, N*N>, R> _matrix{};
    double _largestRowSum = 0.;
    bool _bound = false, _valid = false;
  public:
    /** Computes the weights for these nodes. Returns false if the nodes are not valid. A second call with the
     * same nodes changes nothing. */
    bool bind(const double* nodes) {
      if (nodes == nullptr or not bound::arithmeticSupported()) { _bound = false; return _valid = false; }
      if (_bound and std::equal(_nodes.begin(), _nodes.end(), nodes)) return _valid;
      std::copy_n(nodes, N, _nodes.begin());
      _bound = true;
      _valid = false;
      _largestRowSum = 0.;
      for (int node = 0; node < N; node++)
        if (not std::isfinite(nodes[node]) or nodes[node] < 0. or nodes[node] > 1.
            or (node > 0 and nodes[node] <= nodes[node-1])) return false;
      for (int child = 0; child < R; child++) for (int row = 0; row < N; row++) {
        const Interval coordinate = bound::div(
          bound::add({double(child), double(child)}, {nodes[row], nodes[row]}), double(R));
        for (int col = 0; col < N; col++) {
          Interval weight{1., 1.};
          for (int other = 0; other < N; other++) if (other != col) {
            const auto numerator = bound::sub(coordinate, {nodes[other], nodes[other]});
            const auto denominator = bound::sub({nodes[col], nodes[col]}, {nodes[other], nodes[other]});
            weight = bound::mul(weight, quotient(numerator, denominator));
          }
          if (not bound::valid(weight)) return false;
          _matrix[child][row*N+col] = weight;
        }
        double rowSum = 0.;
        for (int col = 0; col < N; col++)
          rowSum = bound::up(bound::roundedAdd(rowSum, bound::magnitude(_matrix[child][row*N+col])));
        _largestRowSum = std::max(_largestRowSum, rowSum);
      }
      return _valid = std::isfinite(_largestRowSum);
    }
    TARGET_DECLARE
    /** Have the weights been computed? */
    [[nodiscard]] GPU_CALLABLE_METHOD bool valid() const { return _valid; }
    /** The weights for the child at this position along an axis. */
    [[nodiscard]] GPU_CALLABLE_METHOD const std::array<Interval, N*N>& matrix(int child) const { return _matrix[child]; }
    /** The largest sum of the magnitudes of the weights of one child node. Interpolation along one axis
     * enlarges a value by at most this factor. */
    [[nodiscard]] GPU_CALLABLE_METHOD double largestRowSum() const { return _largestRowSum; }
    TARGET_DECLARE_END
  };

  TARGET_DECLARE
  /** Bounds the difference between the values of a child and the parent polynomial at every node of the
   * child, each divided by the scale of its value.
   */
  template<int N, int V, int D, int R, typename P>
  GPU_CALLABLE_METHOD bool residual(const P* fine, const P* parent, const Geometry<N, R>& geometry,
                int child, const std::array<double, V>& scales,
                bound::Field<N, V, D>& output) {
    static_assert(std::is_same_v<P, float> or std::is_same_v<P, double>,
      "Retained-parent certificates require float or double stored values.");
    static_assert(V > 0 and D > 0 and D <= 3);
    if (fine == nullptr or parent == nullptr or not geometry.valid()
        or child < 0 or child >= bound::power(R, D) or not bound::arithmeticSupported()) return false;
    bound::Field<N, V, D> represented;
    std::array<double, V> reference;
    for (int component = 0; component < V; component++) {
      reference[component] = static_cast<double>(parent[component]);
      if (not std::isfinite(reference[component]) or not std::isfinite(scales[component]) or scales[component] <= 0.) return false;
    }
    for (int node = 0; node < bound::power(N, D); node++) for (int component = 0; component < V; component++) {
      auto& value = represented[node*V+component];
      value = bound::difference(static_cast<double>(parent[node*V+component]), reference[component], scales[component]);
      if (not bound::valid(value)) return false;
    }
    for (int axis = 0; axis < D; axis++) {
      represented = bound::contract<N, V, D>(represented, geometry.matrix(child % R), axis);
      child /= R;
    }
    for (int node = 0; node < bound::power(N, D); node++) for (int component = 0; component < V; component++) {
      const auto centeredFine = bound::difference(static_cast<double>(fine[node*V+component]), reference[component], scales[component]);
      output[node*V+component] = bound::sub(centeredFine, represented[node*V+component]);
      if (not bound::valid(output[node*V+component])) return false;
    }
    return true;
  }
  /** A parent polynomial that is prepared once and then compared with each of its children. residual()
   * gives the bounds of the function residual() above. magnitudes() gives a coarser bound at less cost. */
  template<int N, int V, int D, int R, typename P, std::size_t CacheBytes = 32*1024>
  class PreparedParent {
    using Field = bound::Field<N, V, D>;
    /** The parent interpolated along the first axis, for each of the R child positions on that axis. Children
     * with the same position share the result. */
    struct Axis0Cache {
      std::array<Field, R> values;
      std::array<bool, R> ready{};
    };
  public:
    static constexpr bool CachesAxis0 = D > 1 and sizeof(Axis0Cache) <= CacheBytes;
    static constexpr std::size_t CacheStorageBytes = CachesAxis0 ? sizeof(Axis0Cache) : 0;
  private:
    std::conditional_t<CachesAxis0, Axis0Cache, std::array<char, 0>> _axis0;
    Field _normalised;
    std::array<double, V> _reference, _scales, _represented;
    const Geometry<N, R>* _geometry{nullptr};
    bool _valid{false};
  public:
    /** Stores the parent, measured from its first node and divided by the scales, and a bound of the largest
     * value that interpolation can give for each value. */
    GPU_CALLABLE_METHOD bool prepare(const P* parent, const Geometry<N, R>& geometry, const std::array<double, V>& scales) {
      static_assert(std::is_same_v<P, float> or std::is_same_v<P, double>);
      static_assert(N > 0 and V > 0 and D > 0 and D <= 3);
      _valid = false;
      _geometry = &geometry;
      if constexpr (CachesAxis0) _axis0.ready.fill(false);
      if (parent == nullptr or not geometry.valid() or not bound::arithmeticSupported()) return false;
      _scales = scales;
      for (int component = 0; component < V; component++) {
        _reference[component] = static_cast<double>(parent[component]);
        if (not std::isfinite(_reference[component]) or not std::isfinite(scales[component]) or scales[component] <= 0.) return false;
      }
      for (int node = 0; node < bound::power(N, D); node++) for (int component = 0; component < V; component++) {
        auto& value = _normalised[node*V+component];
        value = bound::difference(static_cast<double>(parent[node*V+component]), _reference[component], scales[component]);
        if (not bound::valid(value)) return false;
      }
      // Interpolation along one axis enlarges the largest value by at most the largest row sum. The last
      // factor, slightly above one, covers the rounding of the interpolation.
      for (int component = 0; component < V; component++) {
        double largest = 0.;
        for (int node = 0; node < bound::power(N, D); node++)
          largest = std::max(largest, bound::magnitude(_normalised[node*V+component]));
        for (int axis = 0; axis <= D and largest != 0.; axis++) {
          const double factor = axis < D ? geometry.largestRowSum() : 1. + 0x1p-40;
          largest = std::isfinite(largest) and bound::productSafe(largest, factor)
            ? bound::up(bound::roundedMultiply(largest, factor)) : INFINITY;
        }
        // Adding the smallest normal number covers the rounding of subnormal intermediate values.
        if (largest != 0. and std::isfinite(largest))
          largest = bound::up(bound::roundedAdd(largest, std::numeric_limits<double>::min()));
        _represented[component] = largest;
      }
      return _valid = true;
    }
    /** Bounds, for each value, the largest scaled difference between a child and the parent over the nodes of
     * the child. It does not interpolate the parent, so its bound is coarser than that of residual() and
     * never smaller. Returns false if there is no finite bound.
     */
    GPU_CALLABLE_METHOD bool magnitudes(const P* fine, std::array<double, V>& output) const {
      if (fine == nullptr or not _valid) return false;
      for (int component = 0; component < V; component++) {
        double largest = 0.;
        for (int node = 0; node < bound::power(N, D); node++) {
          const auto centred = bound::difference(static_cast<double>(fine[node*V+component]), _reference[component], _scales[component]);
          if (not bound::valid(centred)) return false;
          largest = std::max(largest, bound::magnitude(centred));
        }
        if (not std::isfinite(_represented[component]) or not bound::sumSafe(largest, _represented[component])) return false;
        // The second step up covers the rounding of the subtraction in residual().
        output[component] = bound::up(bound::up(bound::roundedAdd(largest, _represented[component])));
        if (not std::isfinite(output[component])) return false;
      }
      return true;
    }
    /** Writes bounds of the scaled difference between a child and the parent at every node of the child.
     * child is the position of the child in its parent. */
    GPU_CALLABLE_METHOD bool residual(const P* fine, int child, Field& output) {
      if (fine == nullptr or not _valid or not _geometry->valid()
          or child < 0 or child >= bound::power(R, D) or not bound::arithmeticSupported()) return false;
      Field represented;
      int firstAxis = 0;
      if constexpr (CachesAxis0) {
        const int position = child % R;
        if (not _axis0.ready[position]) {
          _axis0.values[position] = bound::contract<N, V, D>(_normalised, _geometry->matrix(position), 0);
          _axis0.ready[position] = true;
        }
        represented = _axis0.values[position];
        child /= R;
        firstAxis = 1;
      } else represented = _normalised;
      for (int axis = firstAxis; axis < D; axis++) {
        represented = bound::contract<N, V, D>(represented, _geometry->matrix(child % R), axis);
        child /= R;
      }
      for (int node = 0; node < bound::power(N, D); node++) for (int component = 0; component < V; component++) {
        const auto centeredFine = bound::difference(static_cast<double>(fine[node*V+component]), _reference[component], _scales[component]);
        output[node*V+component] = bound::sub(centeredFine, represented[node*V+component]);
        if (not bound::valid(output[node*V+component])) return false;
      }
      return true;
    }
  };
  TARGET_DECLARE_END
} // namespace exahype2::amr::detail::retained_parent
