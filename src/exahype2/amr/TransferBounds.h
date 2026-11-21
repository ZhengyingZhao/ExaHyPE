// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org

#pragma once
#include "tarch/accelerator/CompileTarget.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#if defined(__FAST_MATH__)
#error "Continuous transfer certificates require IEEE arithmetic without fast-math."
#endif
TARGET_DECLARE
namespace exahype2::amr::detail::transfer_bound {
  /** A fixed-capacity list. Appending an entry returns false when the list is full. */
  template <typename T, int Capacity>
  class FixedVector {
    static_assert(Capacity >= 0);
    T _values[Capacity > 0 ? Capacity : 1]{};
    int _size{0};
  public:
    GPU_CALLABLE_METHOD constexpr int size() const { return _size; }
    GPU_CALLABLE_METHOD constexpr bool empty() const { return _size == 0; }
    GPU_CALLABLE_METHOD constexpr T* data() { return _values; }
    GPU_CALLABLE_METHOD constexpr const T* data() const { return _values; }
    GPU_CALLABLE_METHOD constexpr T* begin() { return data(); }
    GPU_CALLABLE_METHOD constexpr const T* begin() const { return data(); }
    GPU_CALLABLE_METHOD constexpr T* end() { return data() + _size; }
    GPU_CALLABLE_METHOD constexpr const T* end() const { return data() + _size; }
    GPU_CALLABLE_METHOD constexpr T& operator[](int index) { return _values[index]; }
    GPU_CALLABLE_METHOD constexpr const T& operator[](int index) const { return _values[index]; }
    GPU_CALLABLE_METHOD constexpr void clear() { _size = 0; }
    GPU_CALLABLE_METHOD constexpr bool resize(int size) {
      if (size < 0 or size > Capacity) return false;
      for (int entry = _size; entry < size; entry++) _values[entry] = T{};
      _size = size;
      return true;
    }
    GPU_CALLABLE_METHOD constexpr bool push_back(const T& value) {
      if (_size == Capacity) return false;
      _values[_size++] = value;
      return true;
    }
  };

  template <typename Iterator>
  GPU_CALLABLE_METHOD constexpr void sort(Iterator first, Iterator last) {
    if (first == last) return;
    for (auto entry = first + 1; entry != last; ++entry) {
      const auto value = *entry;
      auto position = entry;
      while (position != first and value < *(position - 1)) {
        *position = *(position - 1);
        --position;
      }
      *position = value;
    }
  }

  template <typename Iterator>
  GPU_CALLABLE_METHOD constexpr Iterator unique(Iterator first, Iterator last) {
    if (first == last) return last;
    auto output = first;
    for (auto entry = first + 1; entry != last; ++entry) {
      if (not (*output == *entry)) *++output = *entry;
    }
    return output + 1;
  }

  template <typename Iterator>
  GPU_CALLABLE_METHOD constexpr bool nextPermutation(Iterator first, Iterator last) {
    if (first == last or first + 1 == last) return false;
    auto pivot = last - 1;
    while (pivot != first and not (*(pivot - 1) < *pivot)) --pivot;
    const bool found = pivot != first;
    if (found) {
      --pivot;
      auto successor = last - 1;
      while (not (*pivot < *successor)) --successor;
      const auto value = *pivot;
      *pivot = *successor;
      *successor = value;
      ++pivot;
    }
    for (auto back = last - 1; pivot < back; ++pivot, --back) {
      const auto value = *pivot;
      *pivot = *back;
      *back = value;
    }
    return found;
  }

  /** Each operation rounds to double before its result is used to form an interval endpoint. */
  GPU_CALLABLE_METHOD inline double roundedAdd(double left, double right) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(left, right);
#else
    COMPILE_IF_ON_DEVICE(volatile double value = left + right; return value;)
    COMPILE_IF_ON_HOST(return left + right;)
#endif
  }
  GPU_CALLABLE_METHOD inline double roundedSubtract(double left, double right) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(left, right);
#else
    COMPILE_IF_ON_DEVICE(volatile double value = left - right; return value;)
    COMPILE_IF_ON_HOST(return left - right;)
#endif
  }
  GPU_CALLABLE_METHOD inline double roundedMultiply(double left, double right) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(left, right);
#else
    COMPILE_IF_ON_DEVICE(volatile double value = left * right; return value;)
    COMPILE_IF_ON_HOST(return left * right;)
#endif
  }
  GPU_CALLABLE_METHOD inline double roundedDivide(double left, double right) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(left, right);
#else
    COMPILE_IF_ON_DEVICE(volatile double value = left / right; return value;)
    COMPILE_IF_ON_HOST(return left / right;)
#endif
  }

  // Outward-rounded scalar bounds.
  struct Interval {
    double lo, hi;
  };
  GPU_CALLABLE_METHOD inline Interval bad() { return {-INFINITY, INFINITY}; }
  GPU_CALLABLE_METHOD inline bool valid(Interval x) { return std::isfinite(x.lo) && std::isfinite(x.hi) && x.lo <= x.hi; }
  GPU_CALLABLE_METHOD inline bool zero(Interval x) { return x.lo == 0. && x.hi == 0.; }
  // The neighbouring doubles. These return the same bits as std::nextafter without a library call.
  GPU_CALLABLE_METHOD inline double down(double x) {
    if (std::isnan(x) || x == -INFINITY) return x;
    if (x == 0.) return -std::numeric_limits<double>::denorm_min();
    const auto bits = std::bit_cast<std::uint64_t>(x);
    return std::bit_cast<double>(x > 0. ? bits - 1 : bits + 1);
  }
  GPU_CALLABLE_METHOD inline double up(double x) {
    if (std::isnan(x) || x == INFINITY) return x;
    if (x == 0.) return std::numeric_limits<double>::denorm_min();
    const auto bits = std::bit_cast<std::uint64_t>(x);
    return std::bit_cast<double>(x > 0. ? bits + 1 : bits - 1);
  }
  GPU_CALLABLE_METHOD inline double towardsZero(double x) {
    if (std::isnan(x)) return x;
    if (x == 0.) return 0.;
    return std::bit_cast<double>(std::bit_cast<std::uint64_t>(x) - 1);
  }
  // Check for overflow before evaluating interval endpoints.
  GPU_CALLABLE_METHOD inline bool sumSafe(double a, double b) {
    const double limit = std::numeric_limits<double>::max();
    return std::signbit(a) != std::signbit(b) || std::abs(a) <= roundedSubtract(limit, std::abs(b));
  }
  GPU_CALLABLE_METHOD inline bool productSafe(double a, double b) {
    const double aa = std::abs(a), bb = std::abs(b), limit = std::numeric_limits<double>::max();
    if (aa > 1.) {
      // Prevent speculative division by a zero or tiny operand.
      volatile double divisor = aa;
      if (bb > towardsZero(roundedDivide(limit, divisor))) return false;
    }
    if (bb > 1.) {
      volatile double divisor = bb;
      if (aa > towardsZero(roundedDivide(limit, divisor))) return false;
    }
    return true;
  }
  // The result encloses the exact sum, including roundoff at both endpoints.
  GPU_CALLABLE_METHOD inline Interval add(Interval a, Interval b) {
    if (!valid(a) || !valid(b)) {
      return bad();
    }
    if (zero(a)) {
      return b;
    }
    if (zero(b)) {
      return a;
    }
    if (!sumSafe(a.lo, b.lo) || !sumSafe(a.hi, b.hi)) {
      return bad();
    }
    const double lo = roundedAdd(a.lo, b.lo), hi = roundedAdd(a.hi, b.hi);
    return std::isfinite(lo) && std::isfinite(hi) ? Interval{down(lo), up(hi)} : bad();
  }
  GPU_CALLABLE_METHOD inline Interval sub(Interval a, Interval b) { return add(a, {-b.hi, -b.lo}); }
  GPU_CALLABLE_METHOD inline Interval mul(Interval a, Interval b) {
    if (!valid(a) || !valid(b)) {
      return bad();
    }
    if (zero(a) || zero(b)) {
      return {0., 0.};
    }
    // Operands of one sign take their bounds from two known corners. Rounding keeps the order of the
    // four products, so these are the bounds of the general case below.
    const bool positiveA = a.lo >= 0., negativeA = a.hi <= 0., positiveB = b.lo >= 0., negativeB = b.hi <= 0.;
    if ((positiveA || negativeA) && (positiveB || negativeB)) {
      const double lowerA = positiveA == positiveB ? (positiveA ? a.lo : a.hi) : (positiveA ? a.hi : a.lo);
      const double lowerB = positiveA == positiveB ? (positiveA ? b.lo : b.hi) : (positiveA ? b.lo : b.hi);
      const double upperA = positiveA == positiveB ? (positiveA ? a.hi : a.lo) : (positiveA ? a.lo : a.hi);
      const double upperB = positiveA == positiveB ? (positiveA ? b.hi : b.lo) : (positiveA ? b.hi : b.lo);
      // The corner of the largest magnitudes decides whether any product overflows.
      const double largestA = positiveA ? a.hi : a.lo, largestB = positiveB ? b.hi : b.lo;
      if (!productSafe(largestA, largestB)) {
        return bad();
      }
      const double lower = roundedMultiply(lowerA, lowerB), upper = roundedMultiply(upperA, upperB);
      if (!std::isfinite(lower) || !std::isfinite(upper)) {
        return bad();
      }
      return {down(lower), up(upper)};
    }
    if (!productSafe(a.lo, b.lo) || !productSafe(a.lo, b.hi) || !productSafe(a.hi, b.lo) || !productSafe(a.hi, b.hi)) {
      return bad();
    }
    const std::array<double, 4> v{
      roundedMultiply(a.lo, b.lo), roundedMultiply(a.lo, b.hi),
      roundedMultiply(a.hi, b.lo), roundedMultiply(a.hi, b.hi)};
    for (double x : v) {
      if (!std::isfinite(x)) {
        return bad();
      }
    }
    double lower = v[0], upper = v[0];
    for (int entry = 1; entry < 4; entry++) {
      lower = std::min(lower, v[entry]);
      upper = std::max(upper, v[entry]);
    }
    return {down(lower), up(upper)};
  }
  GPU_CALLABLE_METHOD inline Interval div(Interval a, double s) {
    if (!valid(a) || !std::isfinite(s) || s <= 0.) {
      return bad();
    }
    if (zero(a)) {
      return a;
    }
    if (s < 1.) {
      volatile double factor = s;
      if (std::max(std::abs(a.lo), std::abs(a.hi)) > roundedMultiply(std::numeric_limits<double>::max(), factor)) return bad();
    }
    const double lo = roundedDivide(a.lo, s), hi = roundedDivide(a.hi, s);
    return std::isfinite(lo) && std::isfinite(hi) ? Interval{down(lo), up(hi)} : bad();
  }
  // Bound (a-b)/s, dividing first if a-b would overflow.
  GPU_CALLABLE_METHOD inline Interval difference(double a, double b, double s) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(s) || s <= 0.) {
      return bad();
    }
    if (a == b) {
      return {0., 0.};
    }
    const double limit = std::numeric_limits<double>::max();
    if (std::signbit(a) != std::signbit(b) && std::abs(a) > roundedSubtract(limit, std::abs(b))) {
      return sub(div({a, a}, s), div({b, b}, s));
    }
    return div(sub({a, a}, {b, b}), s);
  }
  // Convert an interval to an upper bound on its absolute value.
  GPU_CALLABLE_METHOD inline double magnitude(Interval a) { return valid(a) ? std::max(std::abs(a.lo), std::abs(a.hi)) : INFINITY; }
  // Bound the vector norm with outward rounding; scale by budget to avoid overflow.
  template <int V> GPU_CALLABLE_METHOD bool normPasses(std::array<double, V> m, double budget) {
    if (!std::isfinite(budget) || budget <= 0.) {
      return false;
    }
    sort(m.begin(), m.end());
    double sum = 0.;
    for (double x : m) {
      if (!std::isfinite(x) || x < 0. || x > budget) {
        return false;
      }
      if (x == 0.) {
        continue;
      }
      const double ratio = up(roundedDivide(x, budget)), square = up(roundedMultiply(ratio, ratio));
      sum = up(roundedAdd(sum, square));
      if (!std::isfinite(sum) || sum > 1.) {
        return false;
      }
    }
    return sum == 0. || up(std::sqrt(sum)) <= 1.;
  }
  // Sort terms for an order-independent interval sum.
  template <int N> GPU_CALLABLE_METHOD Interval canonicalSum(std::array<Interval, N> terms) {
    std::array<double, N> lo, hi;
    for (int i = 0; i < N; i++) {
      if (!valid(terms[i])) {
        return bad();
      }
      lo[i] = terms[i].lo;
      hi[i] = terms[i].hi;
    }
    // Negate the lower sum so sign reversal swaps the bounds exactly.
    for (double &x : lo) {
      x = -x;
    }
    sort(lo.begin(), lo.end());
    sort(hi.begin(), hi.end());
    double l = 0., h = 0.;
    for (int i = 0; i < N; i++) {
      if (!sumSafe(l, lo[i]) || !sumSafe(h, hi[i])) {
        return bad();
      }
      if (lo[i] != 0.) {
        l = up(roundedAdd(l, lo[i]));
      }
      if (hi[i] != 0.) {
        h = up(roundedAdd(h, hi[i]));
      }
      if (!std::isfinite(l) || !std::isfinite(h)) {
        return bad();
      }
    }
    return {-l, h};
  }
  // Require round-to-nearest and gradual underflow for interval bounds.
  GPU_CALLABLE_METHOD inline bool arithmeticSupported() {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::digits == 53);
    COMPILE_IF_ON_HOST(if (std::fegetround() != FE_TONEAREST) return false;)
    volatile double normal = std::numeric_limits<double>::min(), half = .5;
    volatile double tiny = std::numeric_limits<double>::denorm_min(), two = 2.;
    const double a = normal * half, b = tiny * two;
    return std::bit_cast<std::uint64_t>(a) == 0x0008000000000000ULL && std::bit_cast<std::uint64_t>(b) == 2ULL;
  }
  GPU_CALLABLE_METHOD constexpr int power(int n, int d) {
    int v = 1;
    while (d--) {
      v *= n;
    }
    return v;
  }
  template <int N, int V, int D> using Field = std::array<Interval, power(N, D) * V>;
  // Apply an interval matrix along one tensor axis.
  template <int N, int V, int D>
  GPU_CALLABLE_METHOD Field<N, V, D> contract(const Field<N, V, D> &q, const std::array<Interval, N * N> &matrix, int axis) {
    Field<N, V, D> out;
    const int stride = power(N, axis), nodes = power(N, D);
    for (int fibre = 0; fibre < nodes / N; fibre++) {
      const int first = (fibre / stride) * (N * stride) + fibre % stride;
      for (int row = 0; row < N; row++) {
        for (int c = 0; c < V; c++) {
          std::array<Interval, N> terms;
          for (int col = 0; col < N; col++) {
            terms[col] = mul(matrix[row * N + col], q[(first + col * stride) * V + c]);
          }
          out[(first + row * stride) * V + c] = canonicalSum<N>(terms);
        }
      }
    }
    return out;
  }
  // Certify the polynomial from the convex hull of its Bernstein control vectors.
  template <int N, int V, int D> GPU_CALLABLE_METHOD bool hullPasses(const Field<N, V, D> &q, double budget) {
    for (int n = 0; n < power(N, D); n++) {
      std::array<double, V> m;
      for (int c = 0; c < V; c++) {
        m[c] = magnitude(q[n * V + c]);
      }
      if (!normPasses<V>(m, budget)) {
        return false;
      }
    }
    return true;
  }
  // Bisect one tensor axis at its midpoint using de Casteljau averaging.
  template <int N, int V, int D> GPU_CALLABLE_METHOD Field<N, V, D> half(const Field<N, V, D> &q, int axis, bool right) {
    Field<N, V, D> out;
    const int stride = power(N, axis), nodes = power(N, D);
    for (int fibre = 0; fibre < nodes / N; fibre++) {
      const int first = (fibre / stride) * (N * stride) + fibre % stride;
      for (int c = 0; c < V; c++) {
        std::array<Interval, N> work, left, back;
        for (int i = 0; i < N; i++) {
          work[i] = q[(first + i * stride) * V + c];
        }
        left[0] = work[0];
        back[N - 1] = work[N - 1];
        for (int depth = 1; depth < N; depth++) {
          for (int i = 0; i < N - depth; i++) {
            work[i] = add(mul(work[i], {.5, .5}), mul(work[i + 1], {.5, .5}));
          }
          left[depth] = work[0];
          back[N - 1 - depth] = work[N - 1 - depth];
        }
        for (int i = 0; i < N; i++) {
          out[(first + i * stride) * V + c] = right ? back[i] : left[i];
        }
      }
    }
    return out;
  }
  /** Bisect all axes until every box passes or the round limit is reached. */
  template <int N, int V, int D, int Remaining>
  GPU_CALLABLE_METHOD bool refinedHullPasses(const Field<N, V, D> &q, double budget, const std::array<int, D> &order) {
    if (hullPasses<N, V, D>(q, budget)) {
      return true;
    }
    if constexpr (Remaining > 0) {
      for (int box = 0; box < (1 << D); box++) {
        auto subbox = q;
        for (int axis : order) {
          subbox = half<N, V, D>(subbox, axis, (box & (1 << axis)) != 0);
        }
        if (!refinedHullPasses<N, V, D, Remaining - 1>(subbox, budget, order)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  // Detect points whose norm lower bound exceeds the budget.
  template <int V> GPU_CALLABLE_METHOD bool pointAbove(const std::array<Interval, V> &point, double budget) {
    std::array<double, V> lower;
    for (int c = 0; c < V; c++) {
      if (!valid(point[c])) {
        return false;
      }
      lower[c] = point[c].lo > 0. ? point[c].lo : point[c].hi < 0. ? -point[c].hi : 0.;
    }
    sort(lower.begin(), lower.end());
    double sum = 0.;
    for (double value : lower) {
      if (value > budget) {
        return true;
      }
      if (value == 0.) {
        continue;
      }
      const double ratio = std::max(0., down(roundedDivide(value, budget))), square = std::max(0., down(roundedMultiply(ratio, ratio)));
      if (!sumSafe(sum, square)) {
        return false;
      }
      sum = std::max(0., down(roundedAdd(sum, square)));
    }
    return sum > 1.;
  }
  // Look for an over-budget value at the corners or centre.
  template <int N, int V, int D>
  GPU_CALLABLE_METHOD bool pointWitness(const Field<N, V, D> &q, double budget, const std::array<int, D> &order) {
    for (int corner = 0; corner < (1 << D); corner++) {
      int node = 0;
      for (int a = 0; a < D; a++) {
        if (corner & (1 << a)) {
          node += (N - 1) * power(N, a);
        }
      }
      std::array<Interval, V> value;
      for (int c = 0; c < V; c++) {
        value[c] = q[node * V + c];
      }
      if (pointAbove<V>(value, budget)) {
        return true;
      }
    }
    // The far corner of the left halves is the original centre.
    auto centre = q;
    for (int a : order) {
      centre = half<N, V, D>(centre, a, false);
    }
    std::array<Interval, V> value;
    for (int c = 0; c < V; c++) {
      value[c] = centre[(power(N, D) - 1) * V + c];
    }
    return pointAbove<V>(value, budget);
  }
  // Split the axis with the largest adjacent-control variation; use order to break ties.
  template <int N, int V, int D> GPU_CALLABLE_METHOD int splitAxis(const Field<N, V, D> &q, const std::array<int, D> &order) {
    int selected = order[0];
    double largest = -1.;
    for (int axis : order) {
      const int stride = power(N, axis), nodes = power(N, D);
      double variation = 0.;
      for (int fibre = 0; fibre < nodes / N; fibre++) {
        const int first = (fibre / stride) * (N * stride) + fibre % stride;
        for (int i = 0; i < N - 1; i++) {
          for (int c = 0; c < V; c++) {
            variation = std::max(variation, magnitude(sub(q[(first + (i + 1) * stride) * V + c],
                                                         q[(first + i * stride) * V + c])));
          }
        }
      }
      if (variation > largest) {
        largest = variation;
        selected = axis;
      }
    }
    return selected;
  }
  enum class HullResult { Proved, Unsafe, Unknown };
  /** Test the error bound on smaller parts of the cell. Unknown means the available bounds or work limit
   * did not settle the test.
   */
  template <int N, int V, int D>
  GPU_CALLABLE_METHOD HullResult adaptiveHullOnDevice(
    const Field<N, V, D>& q, double budget, const std::array<int, D>& order
  ) {
    constexpr int Boxes = (1 << 10) - 1, Coefficients = 1 << 18;
    constexpr int Entries = power(N, D) * V;
    constexpr int Capacity = Entries <= Coefficients ? std::min(512, Coefficients / Entries) : 1;
    if (q.size() > Coefficients) return HullResult::Unknown;
    FixedVector<std::uint64_t, Capacity> active, next;
    active.push_back(0);
    int visited = 0;
    for (int depth = 0;; depth++) {
      if (active.size() > Boxes - visited) return HullResult::Unknown;
      FixedVector<std::uint64_t, Capacity> unresolved;
      for (const auto path : active) {
        auto box = q;
        for (int level = 0; level < depth; level++) {
          const int axis = splitAxis<N, V, D>(box, order);
          box = half<N, V, D>(box, axis, ((path >> (depth - level - 1)) & 1U) != 0);
        }
        ++visited;
        if (hullPasses<N, V, D>(box, budget)) continue;
        for (const auto value : box) if (not valid(value)) return HullResult::Unknown;
        if (pointWitness<N, V, D>(box, budget, order)) return HullResult::Unsafe;
        if (not unresolved.push_back(path)) return HullResult::Unknown;
      }
      if (unresolved.empty()) return HullResult::Proved;
      if (depth >= std::numeric_limits<double>::digits or unresolved.size() > (Boxes - visited) / 2
          or active.size() > Coefficients / int(q.size())
          or unresolved.size() > (Coefficients / int(q.size()) - active.size()) / 2) return HullResult::Unknown;
      next.clear();
      for (const auto path : unresolved) {
        if (not next.push_back(path << 1) or not next.push_back((path << 1) | 1U)) return HullResult::Unknown;
      }
      active = next;
    }
  }

  /** Check whether the error stays within budget throughout the cell. Subdivide boxes whose bounds are
   * inconclusive.
   */
  template <int N, int V, int D>
  GPU_CALLABLE_METHOD HullResult adaptiveHull(const Field<N, V, D> &q, double budget, const std::array<int, D> &order) {
    COMPILE_IF_ON_DEVICE(return (adaptiveHullOnDevice<N, V, D>(q, budget, order));)
    COMPILE_IF_ON_HOST(return ([&]() -> HullResult {
    constexpr std::size_t boxes = (1U << 10) - 1, coefficients = 1U << 18;
    if (q.size() > coefficients) {
      return HullResult::Unknown;
    }
    std::vector<Field<N, V, D>> active{q};
    std::size_t visited = 0;
    for (int depth = 0;; depth++) {
      if (active.size() > boxes - visited) {
        return HullResult::Unknown;
      }
      std::vector<std::size_t> unresolved;
      for (std::size_t index = 0; index < active.size(); index++) {
        const auto &box = active[index];
        visited++;
        if (hullPasses<N, V, D>(box, budget)) {
          continue;
        }
        for (auto value : box) {
          if (!valid(value)) {
            return HullResult::Unknown;
          }
        }
        if (pointWitness<N, V, D>(box, budget, order)) {
          return HullResult::Unsafe;
        }
        unresolved.push_back(index);
      }
      if (unresolved.empty()) {
        return HullResult::Proved;
      }
      // Stop if the whole next frontier would exceed either budget.
      if (depth >= std::numeric_limits<double>::digits || unresolved.size() > (boxes - visited) / 2 ||
          active.size() > coefficients / q.size() ||
          unresolved.size() > (coefficients / q.size() - active.size()) / 2) {
        return HullResult::Unknown;
      }
      std::vector<Field<N, V, D>> next;
      next.reserve(2 * unresolved.size());
      for (std::size_t index : unresolved) {
        const auto &box = active[index];
        const int axis = splitAxis<N, V, D>(box, order);
        next.push_back(half<N, V, D>(box, axis, false));
        next.push_back(half<N, V, D>(box, axis, true));
      }
      active = std::move(next);
    }
    })();)
    return HullResult::Unknown;
  }
  /** Bound the growth of nodal magnitudes under the nodal-to-Bernstein transform of one axis by the
   * largest absolute row sum. Returns false for an invalid weight.
   */
  template <int N>
  GPU_CALLABLE_METHOD bool transformAmplification(const std::array<Interval, N * N> &matrix, double &amplification) {
    amplification = 0.;
    for (int row = 0; row < N; row++) {
      std::array<double, N> terms;
      for (int col = 0; col < N; col++) {
        if (!valid(matrix[row * N + col])) {
          return false;
        }
        terms[col] = magnitude(matrix[row * N + col]);
      }
      sort(terms.begin(), terms.end());
      double sum = 0.;
      for (double x : terms) {
        if (x != 0.) {
          if (!sumSafe(sum, x)) {
            sum = INFINITY;
            break;
          }
          sum = up(roundedAdd(sum, x));
        }
      }
      amplification = std::max(amplification, sum);
    }
    return true;
  }
  /** Return true if nodal magnitudes of at most maxima keep every Bernstein coefficient within budget.
   * Larger maxima never pass where smaller ones fail.
   */
  template <int V, int D>
  GPU_CALLABLE_METHOD bool amplifiedMaximaPass(std::array<double, V> maxima, double budget, double amplification) {
    for (int axis = 0; axis < D; axis++) {
      for (double &x : maxima) {
        if (x != 0.) {
          x = std::isfinite(amplification) && productSafe(x, amplification) ? up(roundedMultiply(x, amplification)) : INFINITY;
        }
      }
    }
    return normPasses<V>(maxima, budget);
  }
  /** Return true if the error intervals at the nodes imply that the polynomial stays within budget
   * everywhere in the cell.
   */
  template <int N, int V, int D>
  GPU_CALLABLE_METHOD bool passesIntervals(const Field<N, V, D> &input, double budget,
                       const std::array<Interval, N * N> &matrix, bool split = true) {
    static_assert(N > 0 && V > 0 && D > 0 && D <= 3);
    static_assert(std::numeric_limits<double>::is_iec559);
    if (!arithmeticSupported()) {
      return false;
    }
    if (!std::isfinite(budget) || budget <= 0.) {
      return false;
    }
    std::array<double, V> maxima{};
    for (int n = 0; n < power(N, D); n++) {
      for (int c = 0; c < V; c++) {
        const auto value = input[n * V + c];
        if (!valid(value)) {
          return false;
        }
        // An over-budget nodal interval prevents certification of the whole box.
        if (magnitude(value) > budget) return false;
        maxima[c] = std::max(maxima[c], magnitude(value));
      }
    }
    // Bound transform amplification by its largest absolute row sum.
    double amplification = 0.;
    if (!transformAmplification<N>(matrix, amplification)) {
      return false;
    }
    if (amplifiedMaximaPass<V, D>(maxima, budget, amplification)) {
      return true;
    }
    // Any contraction order with a passing enclosing hull certifies the bound.
    std::array<int, D> order;
    for (int a = 0; a < D; a++) {
      order[a] = a;
    }
    do {
      auto coefficients = input;
      for (int a : order) {
        coefficients = contract<N, V, D>(coefficients, matrix, a);
      }
      // For constant and multilinear fields every control vector is a corner.
      // Bisection preserves those corners, so a failing hull cannot improve.
      constexpr int UniformRounds = N <= 2 ? 0 : 3;
      const bool proved = split ? refinedHullPasses<N, V, D, UniformRounds>(coefficients, budget, order)
                                : hullPasses<N, V, D>(coefficients, budget);
      if (proved) {
        return true;
      }
    } while (nextPermutation(order.begin(), order.end()));
    // Refine unresolved hulls and reject over-budget point witnesses.
    if constexpr (N <= 2) {
      return false;
    } else if (split) {
      for (int a = 0; a < D; a++) {
        order[a] = a;
      }
      do {
        auto coefficients = input;
        for (int a : order) {
          coefficients = contract<N, V, D>(coefficients, matrix, a);
        }
        const auto result = adaptiveHull<N, V, D>(coefficients, budget, order);
        if (result == HullResult::Proved) {
          return true;
        }
        if (result == HullResult::Unsafe) {
          return false;
        }
      } while (nextPermutation(order.begin(), order.end()));
    }
    return false;
  }

  /** Check that the scaled difference between the original and restored DG polynomials stays within budget
   * across the cell.
   */
  template <int N, int V, int D, typename P>
  GPU_CALLABLE_METHOD bool passes(const P *fine, const P *restored, const std::array<double, V> &scales, double budget,
              const std::array<Interval, N * N> &matrix, bool split = true) {
    if (!fine || !restored || !arithmeticSupported()) {
      return false;
    }
    Field<N, V, D> input;
    for (int n = 0; n < power(N, D); n++) {
      for (int c = 0; c < V; c++) {
        input[n * V + c] = difference(double(fine[n * V + c]), double(restored[n * V + c]), scales[c]);
      }
    }
    return passesIntervals<N, V, D>(input, budget, matrix, split);
  }
} // namespace exahype2::amr::detail::transfer_bound

TARGET_DECLARE_END
