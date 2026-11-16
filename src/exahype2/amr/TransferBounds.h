// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org

#pragma once
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
namespace exahype2::amr::detail::transfer_bound {
  // Outward-rounded scalar bounds.
  struct Interval {
    double lo, hi;
  };
  inline Interval bad() { return {-INFINITY, INFINITY}; }
  inline bool valid(Interval x) { return std::isfinite(x.lo) && std::isfinite(x.hi) && x.lo <= x.hi; }
  inline bool zero(Interval x) { return x.lo == 0. && x.hi == 0.; }
  inline double down(double x) {
    return x == -std::numeric_limits<double>::max() ? -INFINITY : std::nextafter(x, -INFINITY);
  }
  inline double up(double x) {
    return x == std::numeric_limits<double>::max() ? INFINITY : std::nextafter(x, INFINITY);
  }
  // Check for overflow before evaluating interval endpoints.
  inline bool sumSafe(double a, double b) {
    const double limit = std::numeric_limits<double>::max();
    return std::signbit(a) != std::signbit(b) || std::abs(a) <= limit - std::abs(b);
  }
  inline bool productSafe(double a, double b) {
    const double aa = std::abs(a), bb = std::abs(b), limit = std::numeric_limits<double>::max();
    if (aa > 1.) {
      // Prevent speculative division by a zero or tiny operand.
      volatile double divisor = aa;
      if (bb > std::nextafter(limit / divisor, 0.)) return false;
    }
    if (bb > 1.) {
      volatile double divisor = bb;
      if (aa > std::nextafter(limit / divisor, 0.)) return false;
    }
    return true;
  }
  // Round outwards and preserve exact zero; unbounded results fail the certificate.
  inline Interval add(Interval a, Interval b) {
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
    const double lo = a.lo + b.lo, hi = a.hi + b.hi;
    return std::isfinite(lo) && std::isfinite(hi) ? Interval{down(lo), up(hi)} : bad();
  }
  inline Interval sub(Interval a, Interval b) { return add(a, {-b.hi, -b.lo}); }
  inline Interval mul(Interval a, Interval b) {
    if (!valid(a) || !valid(b)) {
      return bad();
    }
    if (zero(a) || zero(b)) {
      return {0., 0.};
    }
    if (!productSafe(a.lo, b.lo) || !productSafe(a.lo, b.hi) || !productSafe(a.hi, b.lo) || !productSafe(a.hi, b.hi)) {
      return bad();
    }
    const std::array<double, 4> v{a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    for (double x : v) {
      if (!std::isfinite(x)) {
        return bad();
      }
    }
    return {down(*std::min_element(v.begin(), v.end())), up(*std::max_element(v.begin(), v.end()))};
  }
  inline Interval div(Interval a, double s) {
    if (!valid(a) || !std::isfinite(s) || s <= 0.) {
      return bad();
    }
    if (zero(a)) {
      return a;
    }
    if (s < 1.) {
      volatile double factor = s;
      if (std::max(std::abs(a.lo), std::abs(a.hi)) > std::numeric_limits<double>::max() * factor) return bad();
    }
    const double lo = a.lo / s, hi = a.hi / s;
    return std::isfinite(lo) && std::isfinite(hi) ? Interval{down(lo), up(hi)} : bad();
  }
  // Bound (a-b)/s, dividing first if a-b would overflow.
  inline Interval difference(double a, double b, double s) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(s) || s <= 0.) {
      return bad();
    }
    if (a == b) {
      return {0., 0.};
    }
    const double limit = std::numeric_limits<double>::max();
    if (std::signbit(a) != std::signbit(b) && std::abs(a) > limit - std::abs(b)) {
      return sub(div({a, a}, s), div({b, b}, s));
    }
    return div(sub({a, a}, {b, b}), s);
  }
  // Convert an interval to an upper bound on its absolute value.
  inline double magnitude(Interval a) { return valid(a) ? std::max(std::abs(a.lo), std::abs(a.hi)) : INFINITY; }
  // Bound the vector norm with outward rounding; scale by budget to avoid overflow.
  template <int V> bool normPasses(std::array<double, V> m, double budget) {
    if (!std::isfinite(budget) || budget <= 0.) {
      return false;
    }
    std::sort(m.begin(), m.end());
    double sum = 0.;
    for (double x : m) {
      if (!std::isfinite(x) || x < 0. || x > budget) {
        return false;
      }
      if (x == 0.) {
        continue;
      }
      const double ratio = up(x / budget), square = up(ratio * ratio);
      sum = up(sum + square);
      if (!std::isfinite(sum) || sum > 1.) {
        return false;
      }
    }
    return sum == 0. || up(std::sqrt(sum)) <= 1.;
  }
  // Sort terms for an order-independent interval sum.
  template <int N> Interval canonicalSum(std::array<Interval, N> terms) {
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
    std::sort(lo.begin(), lo.end());
    std::sort(hi.begin(), hi.end());
    double l = 0., h = 0.;
    for (int i = 0; i < N; i++) {
      if (!sumSafe(l, lo[i]) || !sumSafe(h, hi[i])) {
        return bad();
      }
      if (lo[i] != 0.) {
        l = up(l + lo[i]);
      }
      if (hi[i] != 0.) {
        h = up(h + hi[i]);
      }
      if (!std::isfinite(l) || !std::isfinite(h)) {
        return bad();
      }
    }
    return {-l, h};
  }
  // Require round-to-nearest and gradual underflow for interval bounds.
  inline bool arithmeticSupported() {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::digits == 53);
    if (std::fegetround() != FE_TONEAREST) {
      return false;
    }
    volatile double normal = std::numeric_limits<double>::min(), half = .5;
    volatile double tiny = std::numeric_limits<double>::denorm_min(), two = 2.;
    const double a = normal * half, b = tiny * two;
    return std::bit_cast<std::uint64_t>(a) == 0x0008000000000000ULL && std::bit_cast<std::uint64_t>(b) == 2ULL;
  }
  constexpr int power(int n, int d) {
    int v = 1;
    while (d--) {
      v *= n;
    }
    return v;
  }
  template <int N, int V, int D> using Field = std::array<Interval, power(N, D) * V>;
  // Apply an interval matrix along one tensor axis.
  template <int N, int V, int D>
  Field<N, V, D> contract(const Field<N, V, D> &q, const std::array<Interval, N * N> &matrix, int axis) {
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
  template <int N, int V, int D> bool hullPasses(const Field<N, V, D> &q, double budget) {
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
  template <int N, int V, int D> Field<N, V, D> half(const Field<N, V, D> &q, int axis, bool right) {
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
  template <int N, int V, int D>
  bool refinedHullPasses(const Field<N, V, D> &q, double budget, const std::array<int, D> &order, int remaining) {
    if (hullPasses<N, V, D>(q, budget)) {
      return true;
    }
    if (remaining == 0) {
      return false;
    }
    for (int box = 0; box < (1 << D); box++) {
      auto subbox = q;
      for (int axis : order) {
        subbox = half<N, V, D>(subbox, axis, (box & (1 << axis)) != 0);
      }
      if (!refinedHullPasses<N, V, D>(subbox, budget, order, remaining - 1)) {
        return false;
      }
    }
    return true;
  }
  // Detect points whose norm lower bound exceeds the budget.
  template <int V> bool pointAbove(const std::array<Interval, V> &point, double budget) {
    std::array<double, V> lower;
    for (int c = 0; c < V; c++) {
      if (!valid(point[c])) {
        return false;
      }
      lower[c] = point[c].lo > 0. ? point[c].lo : point[c].hi < 0. ? -point[c].hi : 0.;
    }
    std::sort(lower.begin(), lower.end());
    double sum = 0.;
    for (double value : lower) {
      if (value > budget) {
        return true;
      }
      if (value == 0.) {
        continue;
      }
      const double ratio = std::max(0., down(value / budget)), square = std::max(0., down(ratio * ratio));
      if (!sumSafe(sum, square)) {
        return false;
      }
      sum = std::max(0., down(sum + square));
    }
    return sum > 1.;
  }
  // Look for an over-budget value at the corners or centre.
  template <int N, int V, int D>
  bool pointWitness(const Field<N, V, D> &q, double budget, const std::array<int, D> &order) {
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
  template <int N, int V, int D> int splitAxis(const Field<N, V, D> &q, const std::array<int, D> &order) {
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
  /** Refine unresolved boxes within fixed work limits.
   * Complete frontiers make the limits independent of visit order.
   */
  template <int N, int V, int D>
  HullResult adaptiveHull(const Field<N, V, D> &q, double budget, const std::array<int, D> &order) {
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
  }
  /** Certify a polynomial from interval-enclosed nodal residuals.
   * matrix encloses the nodal-to-Bernstein transform; unresolved bounds fail.
   */
  template <int N, int V, int D>
  bool passesIntervals(const Field<N, V, D> &input, double budget,
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
    for (int row = 0; row < N; row++) {
      std::array<double, N> terms;
      for (int col = 0; col < N; col++) {
        if (!valid(matrix[row * N + col])) {
          return false;
        }
        terms[col] = magnitude(matrix[row * N + col]);
      }
      std::sort(terms.begin(), terms.end());
      double sum = 0.;
      for (double x : terms) {
        if (x != 0.) {
          if (!sumSafe(sum, x)) {
            sum = INFINITY;
            break;
          }
          sum = up(sum + x);
        }
      }
      amplification = std::max(amplification, sum);
    }
    auto fast = maxima;
    for (int axis = 0; axis < D; axis++) {
      for (double &x : fast) {
        if (x != 0.) {
          x = std::isfinite(amplification) && productSafe(x, amplification) ? up(x * amplification) : INFINITY;
        }
      }
    }
    if (normPasses<V>(fast, budget)) {
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
      if (hullPasses<N, V, D>(coefficients, budget)) {
        return true;
      }
      if (split) {
        // Try up to three rounds of uniform bisection before adaptive subdivision.
        bool all = true;
        for (int box = 0; box < (1 << D) && all; box++) {
          auto subbox = coefficients;
          for (int a : order) {
            subbox = half<N, V, D>(subbox, a, (box & (1 << a)) != 0);
          }
          all = refinedHullPasses<N, V, D>(subbox, budget, order, 2);
        }
        if (all) {
          return true;
        }
      }
    } while (std::next_permutation(order.begin(), order.end()));
    // Refine unresolved hulls and reject over-budget point witnesses.
    if (split) {
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
      } while (std::next_permutation(order.begin(), order.end()));
    }
    return false;
  }

  /** Certify the component-scaled fine-minus-restored polynomial over the cell.
   * matrix encloses the nodal-to-Bernstein transform; split enables subdivision.
   */
  template <int N, int V, int D, typename P>
  bool passes(const P *fine, const P *restored, const std::array<double, V> &scales, double budget,
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
