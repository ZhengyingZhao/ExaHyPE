// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "exahype2/amr/AMRMath.h"
#include "exahype2/amr/LimiterEvidence.h"
#include "exahype2/amr/ObservationMetric.h"
#include "exahype2/amr/PredictionSensor.h"
#include "exahype2/amr/RetainedParentBounds.h"
#include "exahype2/dg/limiting/SubcellScheme.h"
#include "tarch/accelerator/Buffer.h"
#include "tarch/accelerator/ParallelFor.h"
#include "tarch/testing/UnitTest.h"
#include "unittests/exahype2/dg/limiting/LimiterTestData.h"

#include <bit>
#include <random>

namespace {
  TARGET_DECLARE
  constexpr int NodesPerAxis   = 2;
  constexpr int MaxPoints      = 3;
  constexpr int Data           = 2;
  constexpr int MaxValues      = peano4::power(MaxPoints, DIMENSIONS) * Data;
  constexpr int ResultsPerCase = 6;

  struct SensorConfig {
    static constexpr int NumberOfObservables = 1;
    static GPU_CALLABLE_METHOD double stateScale(int component) { return component == 0 ? 1.0 : 2.0; }
    static GPU_CALLABLE_METHOD bool resolutionComponent(int component) { return component == 0; }
    static GPU_CALLABLE_METHOD double observableWeight(int, int component) { return component == 0 ? 1.0 : 0.25; }
    static GPU_CALLABLE_METHOD double observableAbsoluteTolerance(int) { return 0.025; }
    static GPU_CALLABLE_METHOD double observableRelativeTolerance(int) { return 0.1; }
  };

  struct InvalidObservationConfig : SensorConfig {
    static GPU_CALLABLE_METHOD double observableWeight(int, int) { return std::numeric_limits<double>::quiet_NaN(); }
  };

  struct SensorCase {
    std::array<double, MaxValues>    own{}, other{};
    std::array<double, NodesPerAxis> nodes{0.21132486540518713, 0.78867513459481287};
    exahype2::amr::NativeCellKey     ownKey{}, otherKey{};
    int                              ownCount{NodesPerAxis}, otherCount{NodesPerAxis};
    bool                             ownFV{false}, otherFV{false};
    int                              normal{0}, side{1};

    GPU_CALLABLE_METHOD exahype2::amr::AcceptedCellView<double> ownView() const {
      return {
        own.data(),
        ownCount,
        ownFV ? exahype2::amr::AcceptedRepresentation::FVCellAverages : exahype2::amr::AcceptedRepresentation::DGNodal
      };
    }
    GPU_CALLABLE_METHOD exahype2::amr::AcceptedCellView<double> otherView() const {
      return {
        other.data(),
        otherCount,
        otherFV ? exahype2::amr::AcceptedRepresentation::FVCellAverages : exahype2::amr::AcceptedRepresentation::DGNodal
      };
    }
  };

  struct EvaluateSensors {
    const SensorCase* cases{nullptr};
    double*           results{nullptr};
    GPU_CALLABLE_METHOD void operator()(int entry) const {
      namespace amr    = exahype2::amr;
      const auto& test = cases[entry];
      const auto  own = test.ownView(), other = test.otherView();
      auto*       result = results + entry * ResultsPerCase;
      result[0] =
        amr::prediction_detail::deviceLocalDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(own, test.nodes.data());
      result[1] = amr::prediction_detail::deviceLocalDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        other,
        test.nodes.data()
      );
      result[2] = amr::prediction_detail::deviceFaceDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      );
      result[3] = amr::resolution_detail::deviceFaceDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      );
      result[4] = amr::observation_detail::deviceFaceRatio<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      );
      result[5] = amr::observation_detail::deviceLocalRatio<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        test.nodes.data(),
        false
      );
    }
  };

  struct RejectInvalidEvidence {
    const SensorCase* cases{nullptr};
    double*           results{nullptr};
    GPU_CALLABLE_METHOD void operator()(int entry) const {
      namespace amr      = exahype2::amr;
      const auto& test   = cases[entry];
      auto        own    = test.ownView();
      const auto  other  = test.otherView();
      auto*       result = results + entry * 4;
      // This patch claims to have more subcells than its storage holds.
      own.pointsPerAxis  = MaxPoints + 1;
      own.representation = amr::AcceptedRepresentation::FVCellAverages;
      result[0] =
        amr::prediction_detail::deviceLocalDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(own, test.nodes.data());
      result[1] = amr::prediction_detail::deviceFaceDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      );
      result[2] = amr::resolution_detail::deviceFaceDetail<NodesPerAxis, Data, SensorConfig, MaxPoints>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      );
      result[3] = amr::observation_detail::deviceLocalRatio<NodesPerAxis, Data, InvalidObservationConfig, MaxPoints>(
        test.ownView(),
        test.nodes.data(),
        false
      );
    }
  };

  struct ParentProofSolver final : exahype2::dg::limiting::unittests::StubSolver {
    static constexpr int NodesPerAxis = 2;
    static constexpr int NumberOfData = 2;
    struct LimiterType {
      using SubcellScheme = exahype2::dg::limiting::SubcellScheme<ParentProofSolver, double>;
      using Detector      = exahype2::dg::limiting::Detector<ParentProofSolver, double>;
    };
    template <typename Precision>
    struct SubcellProjection {
      static GPU_CALLABLE_METHOD constexpr std::array<Precision, 6> getNodesToSubcells1D() {
        return {Precision{0.75}, Precision{0.5}, Precision{0.25}, Precision{0.25}, Precision{0.5}, Precision{0.75}};
      }
      static GPU_CALLABLE_METHOD constexpr std::array<Precision, 4> getNodesToLobatto1D() {
        return {Precision{1}, Precision{0}, Precision{0}, Precision{1}};
      }
    };
  };

  struct EvaluateParentProof {
    int* results{nullptr};
    GPU_CALLABLE_METHOD void operator()(int entry) const {
      constexpr int                                      States = peano4::power(NodesPerAxis, DIMENSIONS);
      std::array<double, States * Data>                  parent{}, child{};
      std::array<const double*, peano4::ChildrenPerCell> children{};
      for (int node = 0; node < States; ++node) {
        parent[node * Data] = child[node * Data] = 1.0;
      }
      for (int childIndex = 0; childIndex < peano4::ChildrenPerCell; ++childIndex) {
        children[childIndex] = child.data();
      }
      if (entry == 1) {
        parent[0] = 2.0;
      }
      if (entry == 2) {
        parent[0] = -1.0;
      }
      const tarch::la::Vector<DIMENSIONS, double> centre(0.5), width(1.0);
      const ParentProofSolver::RuntimeParameters  parameters{};
      results[2 * entry] =
        exahype2::amr::parentPhysicallyAdmissible<ParentProofSolver>(parent.data(), States, centre, width, parameters);
      results[2 * entry + 1] = exahype2::amr::parentRespectsMaximumPrinciple<ParentProofSolver>(
        parent.data(),
        children.data(),
        0.0,
        0.0,
        parameters
      );
    }
  };
  TARGET_DECLARE_END

  /** Pairs DG and FV patches on both sides of a face, with and without a change in mesh level. */
  std::vector<SensorCase> getCases() {
    std::vector<SensorCase> result;
    for (int representations = 0; representations < 4; ++representations) {
      for (int jump = 0; jump < 2; ++jump) {
        for (int normal = 0; normal < DIMENSIONS; ++normal) {
          SensorCase test;
          test.ownFV                = (representations & 1) != 0;
          test.otherFV              = (representations & 2) != 0;
          test.ownCount             = test.ownFV ? MaxPoints : NodesPerAxis;
          test.otherCount           = test.otherFV ? MaxPoints : NodesPerAxis;
          test.ownKey[0]            = 1;
          test.otherKey[0]          = 1 + jump;
          test.otherKey[normal + 1] = jump == 0 ? 1 : peano4::SubdivisionFactor;
          for (int axis = 0; axis < DIMENSIONS; ++axis) {
            if (axis != normal and jump != 0) {
              test.otherKey[axis + 1] = 1;
            }
          }
          test.normal = normal;
          for (int node = 0; node < peano4::power(test.ownCount, DIMENSIONS); ++node) {
            test.own[node * Data]     = 1.0 + 0.07 * (node % 5) + 0.025 * (node % test.ownCount);
            test.own[node * Data + 1] = 3.0 - 0.03125 * (node % 7);
          }
          for (int node = 0; node < peano4::power(test.otherCount, DIMENSIONS); ++node) {
            test.other[node * Data]     = 1.125 - 0.03125 * (node % 7);
            test.other[node * Data + 1] = 2.875 + 0.015625 * (node % 5);
          }
          result.push_back(test);
          std::swap(test.own, test.other);
          std::swap(test.ownCount, test.otherCount);
          std::swap(test.ownFV, test.otherFV);
          std::swap(test.ownKey, test.otherKey);
          test.side = 0;
          result.push_back(test);
        }
      }
    }
    return result;
  }

  std::array<double, ResultsPerCase> getHostReference(const SensorCase& test) {
    namespace amr  = exahype2::amr;
    const auto own = test.ownView(), other = test.otherView();
    return {
      amr::prediction_detail::localDetail<NodesPerAxis, Data, SensorConfig>(own, test.nodes.data()),
      amr::prediction_detail::localDetail<NodesPerAxis, Data, SensorConfig>(other, test.nodes.data()),
      amr::prediction_detail::faceDetail<NodesPerAxis, Data, SensorConfig>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      ),
      amr::resolution_detail::faceDetail<NodesPerAxis, Data, SensorConfig>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      ),
      amr::observation_detail::faceRatio<NodesPerAxis, Data, SensorConfig>(
        own,
        other,
        test.ownKey,
        test.otherKey,
        test.nodes.data(),
        test.normal,
        test.side
      ),
      amr::observation_detail::localRatio<NodesPerAxis, Data, SensorConfig>(own, test.nodes.data(), false)
    };
  }
} // namespace

UNIT_TEST("Device AMR sensors retain FV DG hybrid and coarse fine observation evidence") {
  GIVEN("Nonconstant FV and DG patches in every face orientation at equal and unequal levels") {
    const auto                             cases = getCases();
    tarch::accelerator::Buffer<SensorCase> inputs(tarch::MemoryLocation::Device);
    inputs.setValues(cases);
    tarch::accelerator::Buffer<double> outputs(cases.size() * ResultsPerCase, tarch::MemoryLocation::Device);

    WHEN("A kernel computes local face and observable resolution scores") {
      tarch::accelerator::parallelFor(
        tarch::accelerator::Range(static_cast<int>(cases.size())),
        EvaluateSensors{inputs.getData(), outputs.getData()}
      );
      const auto results = outputs.getValues();

      THEN("All scores match the existing host criterion formulas") {
        for (std::size_t entry = 0; entry < cases.size(); ++entry) {
          const auto expected = getHostReference(cases[entry]);
          for (int score = 0; score < ResultsPerCase; ++score) {
            VALIDATE(std::isfinite(expected[score]), entry, score, expected[score]);
            VALIDATE_NUMERICAL_EQUALS(results[entry * ResultsPerCase + score], expected[score], entry, score);
          }
        }
      }
    }
  }
}

UNIT_TEST("Device AMR sensors reject incomplete storage and invalid observation budgets") {
  GIVEN("A compiled bound smaller than a requested FV layout and nonfinite observation weights") {
    const std::vector<SensorCase>          cases{getCases().front()};
    tarch::accelerator::Buffer<SensorCase> inputs(tarch::MemoryLocation::Device);
    inputs.setValues(cases);
    tarch::accelerator::Buffer<double> outputs(4, tarch::MemoryLocation::Device);

    WHEN("The kernel assesses the invalid inputs") {
      tarch::accelerator::parallelFor(
        tarch::accelerator::Range(1),
        RejectInvalidEvidence{inputs.getData(), outputs.getData()}
      );
      const auto results = outputs.getValues();

      THEN("Every incomplete proof returns infinity") {
        for (const double result : results) {
          VALIDATE(std::isinf(result));
        }
      }
    }
  }
}

UNIT_TEST("Device AMR parent proofs retain physical admissibility and maximum principle vetoes") {
  GIVEN("A constant admissible parent an overshooting parent and a negative parent") {
    tarch::accelerator::Buffer<int> outputs(6, tarch::MemoryLocation::Device);

    WHEN("A kernel proves parent admissibility and observable bounds") {
      tarch::accelerator::parallelFor(tarch::accelerator::Range(3), EvaluateParentProof{outputs.getData()});
      const auto results = outputs.getValues();

      THEN("The constant parent passes and both invalid proposals are vetoed") {
        VALIDATE_EQUALS(results[0], 1);
        VALIDATE_EQUALS(results[1], 1);
        VALIDATE_EQUALS(results[3], 0);
        VALIDATE_EQUALS(results[4], 0);
      }
    }
  }
}

namespace {
  namespace bound = exahype2::amr::detail::transfer_bound;

  /** Fields of both scales measured in units of one and of two. */
  struct ProfileConfig {
    static constexpr double TransferTolerance = 1.0e-3;
    // The sensor and certificate templates that read this configuration are device callable.
    static GPU_CALLABLE_METHOD double stateScale(int component) { return component == 0 ? 1.0 : 2.0; }
    // The Lagrange polynomials of the two Gauss nodes at both ends of the unit interval, enclosed by four ulps.
    static GPU_CALLABLE_METHOD double transferWeight(int node, int mode) {
      constexpr double weights[4]{1.3660254037844386, -0.36602540378443865, -0.36602540378443865, 1.3660254037844386};
      return weights[node * 2 + mode];
    }
    static GPU_CALLABLE_METHOD double transferWeightLower(int node, int mode) {
      double weight = transferWeight(node, mode);
      for (int step = 0; step < 4; ++step) weight = bound::down(weight);
      return weight;
    }
    static GPU_CALLABLE_METHOD double transferWeightUpper(int node, int mode) {
      double weight = transferWeight(node, mode);
      for (int step = 0; step < 4; ++step) weight = bound::up(weight);
      return weight;
    }
  };

  bool haveSameBits(double left, double right) {
    return (std::isnan(left) and std::isnan(right))
        or std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
  }

  /** The interval product from all four corners, as the certificate defined it before the shortcut for signed operands. */
  bound::Interval getFourCornerProduct(bound::Interval a, bound::Interval b) {
    if (not bound::valid(a) or not bound::valid(b)) {
      return bound::bad();
    }
    if (bound::zero(a) or bound::zero(b)) {
      return {0.0, 0.0};
    }
    const std::array<std::array<double, 2>, 4> corners{{{a.lo, b.lo}, {a.lo, b.hi}, {a.hi, b.lo}, {a.hi, b.hi}}};
    double lower = std::numeric_limits<double>::infinity(), upper = -std::numeric_limits<double>::infinity();
    for (const auto& corner : corners) {
      if (not bound::productSafe(corner[0], corner[1])) {
        return bound::bad();
      }
    }
    for (const auto& corner : corners) {
      const double product = bound::roundedMultiply(corner[0], corner[1]);
      if (not std::isfinite(product)) {
        return bound::bad();
      }
      lower = std::min(lower, product);
      upper = std::max(upper, product);
    }
    return {bound::down(lower), bound::up(upper)};
  }
} // namespace

UNIT_TEST("AMR interval bounds step to the neighbouring doubles of the standard library") {
  GIVEN("Zeros, subnormal, extreme and random doubles") {
    const double        infinity = std::numeric_limits<double>::infinity();
    std::vector<double> values{
      0.0, -0.0, 1.0, -1.0, 0.5, infinity, -infinity, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(),
      std::numeric_limits<double>::min(), -std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(), -std::numeric_limits<double>::denorm_min()};
    std::mt19937_64 generator(1);
    for (int sample = 0; sample < 200000; ++sample) {
      values.push_back(std::bit_cast<double>(generator()));
    }

    THEN("Each step returns the bits of std::nextafter") {
      for (const double value : values) {
        VALIDATE(haveSameBits(bound::up(value), std::nextafter(value, infinity)), value);
        VALIDATE(haveSameBits(bound::down(value), std::nextafter(value, -infinity)), value);
        const double towardsZero = std::nextafter(value, 0.0);
        // std::nextafter(-0.0, 0.0) returns the positive zero; the value is the same.
        VALIDATE(std::isnan(towardsZero) ? std::isnan(bound::towardsZero(value)) : bound::towardsZero(value) == towardsZero, value);
      }
    }
  }
}

UNIT_TEST("AMR interval products of one-signed operands equal the bounds from all four corners") {
  GIVEN("Random intervals of either sign, around zero, degenerate, tiny and near overflow") {
    std::mt19937_64                        generator(2);
    std::uniform_real_distribution<double> unit(-1.0, 1.0), exponent(-320.0, 308.0);
    const auto                             draw = [&](int kind) -> bound::Interval {
      const double scale = kind % 3 == 0 ? std::pow(10.0, exponent(generator)) : 1.0;
      double       left = scale * unit(generator), right = scale * unit(generator);
      if (kind % 7 == 1) left = 0.0;
      if (kind % 7 == 2) right = left;
      if (kind % 11 == 3) left = -0.0;
      if (kind % 5 == 4) { left = std::abs(left); right = std::abs(right); }
      if (kind % 5 == 3) { left = -std::abs(left); right = -std::abs(right); }
      return {std::min(left, right), std::max(left, right)};
    };

    THEN("Both products have the same bits or are both unbounded") {
      for (int sample = 0; sample < 200000; ++sample) {
        const auto a = draw(sample), b = draw(sample / 7 + 3 * sample);
        const auto product = bound::mul(a, b), expected = getFourCornerProduct(a, b);
        VALIDATE(haveSameBits(product.lo, expected.lo), a.lo, a.hi, b.lo, b.hi, product.lo, expected.lo);
        VALIDATE(haveSameBits(product.hi, expected.hi), a.lo, a.hi, b.lo, b.hi, product.hi, expected.hi);
      }
    }
  }
}

UNIT_TEST("AMR DG profiles on tangential nodes skip interpolation without changing a bit") {
  GIVEN("Random DG cells, constant cells and cells with zeros, sampled on nodes, beside nodes and between nodes") {
    namespace amr       = exahype2::amr;
    constexpr int Nodes = peano4::power(NodesPerAxis, DIMENSIONS);
    const std::array<double, NodesPerAxis> nodes{0.21132486540518713, 0.78867513459481287};
    std::mt19937_64                        generator(3);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);

    THEN("The cached host profile, the direct host profile and the device profile agree") {
      for (int sample = 0; sample < 4000; ++sample) {
        std::array<double, Nodes * Data> values{};
        for (double& value : values) {
          value = sample % 5 == 0 ? 0.75 : (sample % 7 == 0 and generator() % 2 == 0 ? 0.0 : unit(generator));
        }
        const std::array<double, Data> reference{unit(generator), sample % 3 == 0 ? values[1] : unit(generator)};
        const int                      normal = sample % DIMENSIONS;
        std::array<double, DIMENSIONS> point{};
        bool                           onNodes = true;
        for (int axis = 0; axis < DIMENSIONS; ++axis) {
          const int kind = (sample / (axis + 1)) % 4;
          point[axis]    = nodes[(sample + axis) % NodesPerAxis];
          // A conforming face maps its samples onto the nodes up to the last bits.
          if (kind == 1) point[axis] = std::nextafter(point[axis], 1.0);
          if (kind == 2) point[axis] = std::nextafter(std::nextafter(point[axis], 0.0), 0.0);
          if (kind == 3) point[axis] = 0.5 + 0.5 * unit(generator);
          onNodes = onNodes and (axis == normal or kind != 3);
        }
        const amr::AcceptedCellView<double> view{values.data(), NodesPerAxis, amr::AcceptedRepresentation::DGNodal};
        const amr::prediction_detail::DGProfileGeometry<NodesPerAxis> geometry(nodes.data(), normal, point);
        VALIDATE(geometry.valid, sample);
        VALIDATE_EQUALS(geometry.single >= 0, onNodes, sample);

        std::array<double, NodesPerAxis * Data> cached{}, direct{}, device{};
        amr::prediction_detail::DGSourceProfileCache<NodesPerAxis, Data, ProfileConfig, double> source(view, normal, reference);
        VALIDATE(source.profile(nodes.data(), point, cached, 0), sample);
        const bool directIsValid = amr::prediction_detail::dgProfile<NodesPerAxis, Data, ProfileConfig>(
          view, nodes.data(), normal, point, reference, direct, 0
        );
        const bool deviceIsValid = amr::prediction_detail::deviceDGProfile<NodesPerAxis, Data, ProfileConfig>(
          view, nodes.data(), normal, point, reference, device, 0
        );
        VALIDATE(directIsValid, sample);
        VALIDATE(deviceIsValid, sample);
        for (int entry = 0; entry < NodesPerAxis * Data; ++entry) {
          VALIDATE(haveSameBits(cached[entry], device[entry]), sample, entry, cached[entry], device[entry]);
          VALIDATE(haveSameBits(direct[entry], device[entry]), sample, entry, direct[entry], device[entry]);
        }
      }
    }
  }
}

UNIT_TEST("AMR magnitude bounds never accept a transfer that the interval certificate rejects") {
  GIVEN("Random parents and children from subnormal to huge amplitudes around zero and around large offsets") {
    namespace retained  = exahype2::amr::detail::retained_parent;
    constexpr int Nodes = peano4::power(NodesPerAxis, DIMENSIONS);
    const std::array<double, NodesPerAxis>                        nodes{0.21132486540518713, 0.78867513459481287};
    retained::Geometry<NodesPerAxis, peano4::SubdivisionFactor>   geometry;
    VALIDATE(geometry.bind(nodes.data()));
    const std::array<double, Data>         scales{ProfileConfig::stateScale(0), ProfileConfig::stateScale(1)};
    std::mt19937_64                        generator(4);
    std::uniform_real_distribution<double> unit(-1.0, 1.0), wide(-330.0, 300.0), narrow(-12.0, -1.0);
    int                                    accepted = 0, compared = 0;

    THEN("Each bound covers the evaluated residual and each accepted bound has a passing certificate") {
      for (int sample = 0; sample < 40000; ++sample) {
        const int    regime = sample % 4;
        const double level  = regime == 0 ? std::pow(10.0, wide(generator)) : std::pow(10.0, narrow(generator)) * (regime == 3 ? 1.0e-3 : 1.0);
        const double offset = regime == 2 ? 1.0 : (regime == 1 ? 1.0e3 * unit(generator) : 0.0);
        std::array<double, Nodes * Data> parent{}, child{};
        for (int entry = 0; entry < Nodes * Data; ++entry) {
          parent[entry] = (offset + level * unit(generator)) * scales[entry % Data];
          child[entry]  = (offset + level * unit(generator)) * scales[entry % Data];
        }
        retained::PreparedParent<NodesPerAxis, Data, DIMENSIONS, peano4::SubdivisionFactor, double> prepared;
        if (not prepared.prepare(parent.data(), geometry, scales)) {
          continue;
        }
        std::array<double, Data> magnitudes{};
        if (not prepared.magnitudes(child.data(), magnitudes)) {
          continue;
        }
        bound::Field<NodesPerAxis, Data, DIMENSIONS> residual;
        const int position = static_cast<int>(generator() % peano4::ChildrenPerCell);
        VALIDATE(prepared.residual(child.data(), position, residual), sample);
        ++compared;
        for (int node = 0; node < Nodes; ++node) {
          for (int component = 0; component < Data; ++component) {
            VALIDATE(bound::magnitude(residual[node * Data + component]) <= magnitudes[component], sample, node, component);
          }
        }
        if (exahype2::amr::amr_detail::transferMagnitudesPass<NodesPerAxis, Data, ProfileConfig>(magnitudes)) {
          ++accepted;
          const bool certified =
            exahype2::amr::amr_detail::transferPolynomialIntervalsPasses<NodesPerAxis, Data, ProfileConfig>(residual);
          VALIDATE(certified, sample);
        }
      }
      // The test is void unless both outcomes of the shortcut occur.
      VALIDATE(accepted > 1000, accepted);
      VALIDATE(compared - accepted > 1000, compared, accepted);
    }
  }
}

namespace {
  /** The run-time recursion that the certificate replaces by one instantiation per round. */
  template <int N, int V, int D>
  bool passesRecursiveHullRefinement(
    const bound::Field<N, V, D>& box, double budget, const std::array<int, D>& order, int remaining
  ) {
    if (bound::hullPasses<N, V, D>(box, budget)) {
      return true;
    }
    if (remaining == 0) {
      return false;
    }
    for (int half = 0; half < (1 << D); ++half) {
      auto halfBox = box;
      for (const int axis : order) {
        halfBox = bound::half<N, V, D>(halfBox, axis, (half & (1 << axis)) != 0);
      }
      if (not passesRecursiveHullRefinement<N, V, D>(halfBox, budget, order, remaining - 1)) {
        return false;
      }
    }
    return true;
  }
} // namespace

UNIT_TEST("AMR hull refinement without recursion keeps the verdict of the recursive bisection") {
  GIVEN("Random coefficient boxes around the budget for every axis order") {
    constexpr int   Degree = 3;
    constexpr int   Nodes  = peano4::power(Degree, DIMENSIONS);
    constexpr double Budget = 1.0e-3;
    std::mt19937_64                        generator(5);
    std::uniform_real_distribution<double> unit(-1.0, 1.0), level(0.2, 4.0);
    int                                    passed = 0, failed = 0;

    THEN("Both evaluations agree after zero, one and two rounds") {
      for (int sample = 0; sample < 3000; ++sample) {
        bound::Field<Degree, Data, DIMENSIONS> box;
        const double                           amplitude = Budget * level(generator);
        // A bump along each axis has a peak coefficient above its values, so the hull tightens under bisection.
        std::array<std::array<double, Degree>, DIMENSIONS> profile;
        for (auto& axis : profile) {
          axis = {0.5 + 0.05 * unit(generator), 1.2 + 0.1 * unit(generator), 0.5 + 0.05 * unit(generator)};
        }
        for (int node = 0; node < Nodes; ++node) {
          double value = amplitude;
          for (int axis = 0, index = node; axis < DIMENSIONS; ++axis, index /= Degree) {
            value *= profile[axis][index % Degree];
          }
          for (int component = 0; component < Data; ++component) {
            const double scaled        = component == 0 ? value : 0.3 * unit(generator) * value;
            box[node * Data + component] = {bound::down(scaled), bound::up(scaled)};
          }
        }
        std::array<int, DIMENSIONS> order;
        for (int axis = 0; axis < DIMENSIONS; ++axis) {
          order[axis] = (axis + sample) % DIMENSIONS;
        }
        const bool none = bound::refinedHullPasses<Degree, Data, DIMENSIONS, 0>(box, Budget, order);
        const bool once = bound::refinedHullPasses<Degree, Data, DIMENSIONS, 1>(box, Budget, order);
        const bool twice = bound::refinedHullPasses<Degree, Data, DIMENSIONS, 2>(box, Budget, order);
        const std::array<bool, 3> expected{
          passesRecursiveHullRefinement<Degree, Data, DIMENSIONS>(box, Budget, order, 0),
          passesRecursiveHullRefinement<Degree, Data, DIMENSIONS>(box, Budget, order, 1),
          passesRecursiveHullRefinement<Degree, Data, DIMENSIONS>(box, Budget, order, 2)};
        VALIDATE_EQUALS(none, expected[0], sample);
        VALIDATE_EQUALS(once, expected[1], sample);
        VALIDATE_EQUALS(twice, expected[2], sample);
        passed += twice and not none ? 1 : 0;
        failed += twice ? 0 : 1;
      }
      // Both verdicts occur, and some boxes pass only after bisection.
      VALIDATE(passed > 20, passed);
      VALIDATE(failed > 20, failed);
    }
  }
}
