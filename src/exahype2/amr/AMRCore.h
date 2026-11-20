// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "AMRPolicies.h"
#include "RetainedParentBounds.h"
#include "ResolutionSensor.h"
#include "PredictionSensor.h"
#include "ObservationMetric.h"
#include "exahype2/RefinementCommand.h"
#include "exahype2/TimeStepping.h"
#include "peano4/CellConnectivity.h"
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace exahype2::amr {
  /** Assess refinement and coarsening using accepted solution data. */
  template<int NodesPerAxis, int NumberOfData, typename Config, typename Precision,
           typename Accepted, typename Transfer, typename Evidence = NoLimiterEvidence,
           typename Assessment = DefaultAssessmentPolicy<Accepted>>
  class AMRCore final {
    template<int, int, typename, typename, typename, typename, typename, typename>
    friend class DeviceCriterion;
  public:
    using EngineTag = AMRCoreTag;
    using AcceptedAdapter = Accepted;
    using TransferPolicy = Transfer;
    using StabilityEvidence = Evidence;
    using LimiterEvidence = Evidence;
    using AssessmentPolicy = Assessment;
    using Coordinates = tarch::la::Vector<DIMENSIONS, double>;
    // Own the callbacks and their captured solver parameters.
    using WaveSpeed = std::function<double(const Precision*, const Coordinates&, const Coordinates&, double, double, int)>;
    using PhysicalAdmissibility = std::function<bool(const Precision*, int, const Coordinates&, const Coordinates&)>;
    using InitialDataCheck = std::function<bool(const Precision*, const Coordinates&, const Coordinates&)>;
    using TransferAdmissibility = std::function<bool(const Precision*, const Precision* const*)>;
    using NativeSubcells = typename tarch::accelerator::ChunkedMemoryPool<Precision>::ConstView;

    /** Counters for cached assessments and refinement votes. */
    struct Statistics {
      std::size_t totalFamilies{0};
      std::size_t completeFamilies{0};
      std::size_t assessedFamilies{0};
      /** Families exceeding the maximum allowed parent size. */
      std::size_t sizeVetoedFamilies{0};
      std::size_t transferPassed{0};
      std::size_t currentNeighboursPassed{0};
      std::size_t eligibleFamilies{0};
      /** Alias for eligibleFamilies. */
      std::size_t quietReady{0};
      /** Reserved for statistics compatibility. */
      std::size_t budgetReady{0};
      std::size_t eraseVotes{0};
      std::size_t conflictVetoes{0};
      std::size_t assessedLeaves{0};
      std::size_t refineEvidence{0};
    };

    void beginInitialisation(PhysicalAdmissibility admissible = nullptr, InitialDataCheck initialDataCheck = nullptr);
    /** Sets up host assessment for this time step using the solver's cells, solution and optional limiter data. */
    template<typename SolutionView, typename StatusView>
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution, StatusView statuses,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed = nullptr,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr,
                   NativeSubcells subcells = {});
    [[nodiscard]] exahype2::RefinementCommand command(
      const Precision* Q, const Coordinates& x, const Coordinates& h, double time) const;
    void finishStep();
    /** Counts assessed leaves and sibling groups. Call after voting has finished.
     * One erase vote represents a whole group of siblings that can coarsen together. */
    [[nodiscard]] Statistics statistics() const;
    void setReferenceTime(double time);
    void reset();

  private:
    static_assert(NodesPerAxis > 0 and NumberOfData > 0);
    static constexpr bool IsFiniteVolume = TransferPolicy::IsFiniteVolume;
    static constexpr int Nodes = peano4::power(NodesPerAxis, DIMENSIONS);
    static constexpr int Values = Nodes * NumberOfData;
    [[nodiscard]] static constexpr bool predictionMode() {
      if constexpr (requires { Config::UsePredictionCriterion; }) return Config::UsePredictionCriterion;
      return false;
    }
    [[nodiscard]] static constexpr bool cacheAcceptedFaces() {
      if constexpr (requires { Config::CacheAcceptedFaces; })
        return predictionMode() and Config::CacheAcceptedFaces;
      return predictionMode();
    }
    /** Convert an observable's budget ratio to resolution-score units. */
    [[nodiscard]] static double withObservation(double detail, double ratio);
    [[nodiscard]] static double acceptedLocalDetail(const AcceptedCellView<Precision>& view, const double* nodes);
    using Key = std::array<int, DIMENSIONS + 1>;
    using Geometry = std::array<double, 2 * DIMENSIONS>;
    struct History {
      double time{0.0};
      double quiet{0.0};
      bool live{false};
    };
    struct Cell {
      Key key{};
      int slot{-1};
      Coordinates centre{}, width{};
      const Precision* state{nullptr};
      const int* status{nullptr};
      History* history{nullptr};
      double quiet{0.0};
      mutable double acceptedQuiet{0.0};
      mutable double crossing{0.0};
      mutable bool moving{false};
      mutable bool quietReady{false};
      int family{-1};
    };
    /** Resolution score and completeness of the neighbour data. */
    struct ResolutionEvidence {
      double value{0.0};
      bool complete{true};
    };
    struct DetailCache {
      std::atomic<int> state{0};
      ResolutionEvidence evidence{};
    };
    struct NativeFace {
      int slot{-1}, normal{-1}, side{-1}, cache{-1};
    };
    struct NativeStencil {
      std::vector<NativeFace> faces;
      bool complete{false};
    };
    /** Share each accepted face score across adjacent cells within one step. */
    struct AcceptedFaceCache {
      std::atomic<int> state{0};
      int left{-1}, right{-1}, normal{-1};
      double value{0.0};
    };
    struct Family {
      std::array<int, peano4::ChildrenPerCell> slots{};
      bool complete{false};
    };
    /** Store transfer and accepted-neighbour checks for a proposed parent. */
    struct FamilyPlan {
      std::atomic<int> state{0};
      std::array<Precision, Values> parent{};
      Key key{};
      bool sizeVetoed{false};
      bool transferSafe{false};
      bool neighboursSafe{false};
      bool safe{false};
      bool eligible{false};
    };
    /** Keep a region's quiet history when its children coarsen. */
    struct QuietRegion {
      History* history{nullptr};
      std::array<int, peano4::ChildrenPerCell> children{};
      int slot{-1};
      double crossing{0.0};
      bool moving{false};
      bool valid{false};
    };

    mutable std::unique_ptr<DetailCache[]> _detailCache;
    std::size_t _detailCacheCapacity{0};
    std::vector<NativeStencil> _nativeStencils;
    mutable std::unique_ptr<AcceptedFaceCache[]> _acceptedFaces;
    std::size_t _acceptedFaceCount{0}, _acceptedFaceCapacity{0};
    std::map<Key, History> _history;
    std::map<Geometry, int> _lookup;
    std::vector<Cell> _cells;
    std::vector<int> _leaves;
    std::vector<Family> _families;
    std::vector<QuietRegion> _quietRegions;
    std::vector<int> _quietOrder;
    mutable std::unique_ptr<std::atomic<int>[]> _familyVotes;
    mutable std::unique_ptr<FamilyPlan[]> _familyPlans;
    std::size_t _familyVoteCapacity{0};
    const exahype2::TimeStepping* _clock{nullptr};
    const double* _nodes{nullptr};
    [[no_unique_address]] std::conditional_t<IsFiniteVolume, detail::retained_parent::NoGeometry,
      detail::retained_parent::Geometry<NodesPerAxis, peano4::SubdivisionFactor>> _retainedParentGeometry;
    const double* _restriction{nullptr};
    const double* _prolongation{nullptr};
    WaveSpeed _waveSpeed{nullptr};
    PhysicalAdmissibility _admissible{nullptr};
    InitialDataCheck _initialDataCheck{nullptr};
    TransferAdmissibility _transferAdmissible{nullptr};
    const peano4::CellConnectivity* _mesh{nullptr};
    NativeSubcells _subcells{};
    double _startTime{0.0};
    double _referenceTime{[] {
      if constexpr (predictionMode()) return 0.0;
      else return Config::ReferenceTime;
    }()};
    double _restrictionBound{0.0};
    double _prolongationBound{0.0};
    bool _active{false};
    bool _initialising{false};
    bool _topologyCached{false};
    bool _historyNeedsPruning{false};

    [[nodiscard]] static Geometry geometry(const Coordinates& x, const Coordinates& h);
    [[nodiscard]] static double amplification(const double* matrix, bool restriction);
    [[nodiscard]] static bool finite(const Precision* Q);
    [[nodiscard]] static bool finite(const AcceptedCellView<Precision>& view);
    [[nodiscard]] AcceptedCellView<Precision> accepted(const Cell& cell) const;
    [[nodiscard]] bool locallyConstant(const Cell& cell) const;
    [[nodiscard]] static double localDetail(const Precision* Q);
    [[nodiscard]] static double localDetail(const AcceptedCellView<Precision>& view);
    [[nodiscard]] ResolutionEvidence cellDetail(const Cell& cell) const;
    [[nodiscard]] ResolutionEvidence computeCellDetail(const Cell& cell) const;
    void bindAcceptedFaces(const peano4::CellConnectivity& mesh);
    [[nodiscard]] double acceptedFaceDetail(int face) const;
    [[nodiscard]] double faceDetail(
      const AcceptedCellView<Precision>& own, const AcceptedCellView<Precision>& other,
      const Key& ownKey, const Key& otherKey, int normal, int side
    ) const;
    /** Sets up the grid geometry, sibling groups and transfer bounds for CPU or device assessment.
     * Returns true if the step starts at a finite, nonnegative time. */
    [[nodiscard]] bool prepareStep(const peano4::CellConnectivity& mesh, const exahype2::TimeStepping& clock,
                                   const double* nodes, const double* restriction, const double* prolongation);
    static void countAssessedFamily(Statistics& result, const auto& plan) {
      ++result.assessedFamilies;
      result.sizeVetoedFamilies += plan.sizeVetoed;
      result.transferPassed += plan.transferSafe;
      result.currentNeighboursPassed += plan.neighboursSafe;
      result.eligibleFamilies += plan.eligible;
      result.quietReady += plan.eligible;
    }
    void bindQuietRegions();
    void finishQuietRegions(double time, double dt);
    [[nodiscard]] double advanceQuiet(double quiet, double crossing, bool moving, double dt) const;
    [[nodiscard]] bool fresh(double time) const;
    [[nodiscard]] double quietProgress(const Cell& cell, double time, double dt) const;
    [[nodiscard]] double computeQuietProgress(const Cell& cell, double time, double dt) const;
    [[nodiscard]] bool familyPasses(int family, double time, double dt) const;
    [[nodiscard]] const FamilyPlan& familyPlan(int family, double time, double dt) const;
    [[nodiscard]] bool transferPasses(
      const Family& family, double time, double dt, Precision* parent, Key& parentKey
    ) const;
    [[nodiscard]] bool parentPassesNeighbours(
      const Family& family, const Precision* parent, const Key& parentKey,
      double time, double dt, bool onlyProposedNeighbours
    ) const;
  };
} // namespace exahype2::amr
#include "AMRCore.cpph"
