// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "exahype2/amr/LimiterEvidence.h"
#include "exahype2/amr/ObservationMetric.h"
#include "exahype2/dg/limiting/SubcellScheme.h"
#include "tarch/accelerator/Buffer.h"
#include "tarch/accelerator/ParallelFor.h"
#include "tarch/testing/UnitTest.h"
#include "unittests/exahype2/dg/limiting/LimiterTestData.h"

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
