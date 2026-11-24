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
  /**
   * Decides for every leaf cell of one solver whether it refines, keeps its size or coarsens.
   *
   * The solver calls beginStep() before a time step and command() for every cell after it. The
   * decision uses the solution of the completed step. The comments call it the accepted solution.
   *
   * A cell refines if its score exceeds the resolution tolerance. The score measures how poorly
   * the cell and its faces resolve the solution. The code calls it the detail. A cell also
   * refines if its solution is not physically admissible or if the limiter asks for it.
   *
   * A cell coarsens only together with its siblings. The comments call the siblings a family and
   * the coarse cell that would replace them the proposed parent. A family coarsens if
   *  - the parent reproduces the children within the transfer tolerance,
   *  - the score of the parent stays within the resolution tolerance against its neighbours, and
   *  - the same holds against neighbouring parents that are proposed in the same step.
   *
   * The score comes from PredictionSensor.h in prediction mode and from ResolutionSensor.h in
   * legacy mode. In legacy mode a family also has to stay resolved for a while before it
   * coarsens. This quiet time is counted in the time a wave takes to cross a cell.
   *
   * NodesPerAxis is the number of DG nodes or FV volumes per axis and NumberOfData the number of
   * values per node. Config holds the tolerances and the scale of every value. Accepted reads the
   * accepted solution of a cell, Transfer moves data between a parent and its children, Evidence
   * tells what a limiter knows about a cell and Assessment selects the checks. AMRPolicies.h and
   * AssessmentPolicy.h hold the choices.
   */
  template<int NodesPerAxis, int NumberOfData, typename Config, typename Precision,
           typename Accepted, typename Transfer, typename Evidence = NoLimiterEvidence,
           typename Assessment = DefaultAssessmentPolicy<Accepted>>
  class AMRCore final {
    template<int, int, typename, typename, typename, typename, typename, typename>
    friend class DeviceCriterion;
  public:
    using AcceptedAdapter = Accepted;
    using TransferPolicy = Transfer;
    using LimiterEvidence = Evidence;
    using AssessmentPolicy = Assessment;
    using Coordinates = tarch::la::Vector<DIMENSIONS, double>;
    // The callbacks into the solver. Each one keeps the solver parameters that it captured.
    /** The largest wave speed of one state along one axis: (state, position, cell width, time, step size, axis). */
    using WaveSpeed = std::function<double(const Precision*, const Coordinates&, const Coordinates&, double, double, int)>;
    /** Are these states physically admissible? (states, their number, cell centre, cell width) */
    using PhysicalAdmissibility = std::function<bool(const Precision*, int, const Coordinates&, const Coordinates&)>;
    /** Does this cell represent the initial condition well enough? (cell values, cell centre, cell width) */
    using InitialDataCheck = std::function<bool(const Precision*, const Coordinates&, const Coordinates&)>;
    /** May this parent replace these children? (parent values, values of every child) */
    using TransferAdmissibility = std::function<bool(const Precision*, const Precision* const*)>;
    /** The FV subcell values that a subcell limiter stores. */
    using NativeSubcells = typename tarch::accelerator::ChunkedMemoryPool<Precision>::ConstView;

    /** What the criterion counted in the current step. */
    struct Statistics {
      /** Families in the mesh. */
      std::size_t totalFamilies{0};
      /** Families whose cells are all leaves. Only these can coarsen. */
      std::size_t completeFamilies{0};
      /** Families that were checked for coarsening. */
      std::size_t assessedFamilies{0};
      /** Checked families whose parent would be larger than the mesh allows. */
      std::size_t sizeVetoedFamilies{0};
      /** Checked families whose parent reproduces the children within the transfer tolerance. */
      std::size_t transferPassed{0};
      /** Checked families whose parent also fits its current neighbours. */
      std::size_t currentNeighboursPassed{0};
      /** Checked families that passed every check of their own. */
      std::size_t eligibleFamilies{0};
      /** The same number as eligibleFamilies. */
      std::size_t quietReady{0};
      /** Families that voted to coarsen. */
      std::size_t eraseVotes{0};
      /** Families that passed their own checks but do not fit a neighbouring family that coarsens too. */
      std::size_t conflictVetoes{0};
      /** Leaves whose score was computed. */
      std::size_t assessedLeaves{0};
      /** Leaves whose score exceeds the resolution tolerance. */
      std::size_t refineEvidence{0};
      /** Votes that kept a cell because its position or solution was not the one given to beginStep(). */
      std::size_t unboundVotes{0};
    };

    /** Starts the votes on the initial mesh. During initialisation a cell refines or keeps its size. */
    void beginInitialisation(PhysicalAdmissibility admissible = nullptr, InitialDataCheck initialDataCheck = nullptr);
    /** Gives the initial votes the mesh and the initial data of every cell, so that they compare neighbours as
     * the votes of a time step do. Has no effect outside initialisation or in legacy mode. */
    template<typename SolutionView, typename StatusView>
    void bindInitialState(const peano4::CellConnectivity& mesh, SolutionView solution, StatusView statuses,
                          const double* nodes);
    /** Sets up host assessment for this time step using the solver's cells, solution and optional limiter data. */
    template<typename SolutionView, typename StatusView>
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution, StatusView statuses,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed = nullptr,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr,
                   NativeSubcells subcells = {});
    /** Evaluates this step's evidence in parallel stages. command() then only reads the results.
     * Call it once the accepted solution is in the storage bound by beginStep().
     * The votes are the same without this call, but concurrent command() calls then wait for each other. */
    void assess(double time);
    /** Selects whether this step assesses coarsening. Refinement is assessed on every bound step. */
    void setCoarseningAssessment(bool enabled);
    /** Returns the vote of the cell at x with width h: Refine, Keep or Erase. Q is the solution of the cell
     * that beginStep() received and time the end of the step. Erase means that the whole family can coarsen. */
    [[nodiscard]] exahype2::RefinementCommand command(
      const Precision* Q, const Coordinates& x, const Coordinates& h, double time) const;
    /** Ends the step. In legacy mode it stores the quiet time of every cell for the next step. */
    void finishStep();
    /** Counts assessed leaves and sibling groups. Call after voting has finished.
     * One erase vote represents a whole group of siblings that can coarsen together. */
    [[nodiscard]] Statistics statistics() const;
    /** Sets the crossing time that legacy mode uses for a cell in which nothing moves. */
    void setReferenceTime(double time);
    /** Forgets the mesh, the solver callbacks and the quiet times. */
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
    /** Combines the score of the values with that of the observables and returns the larger one. ratio is
     * the largest error of an observable divided by its tolerance. A ratio above one gives a score above the
     * resolution tolerance. */
    [[nodiscard]] static double withObservation(double detail, double ratio);
    /** Returns the score of one cell without its faces. */
    [[nodiscard]] static double acceptedLocalDetail(const AcceptedCellView<Precision>& view, const double* nodes);
    /** The level of a cell and its integer position on that level. */
    using Key = std::array<int, DIMENSIONS + 1>;
    /** The centre and the width of a cell. */
    using Geometry = std::array<double, 2 * DIMENSIONS>;
    /** The quiet time of a cell or region (legacy mode) and the time at which it was stored. */
    struct History {
      double time{0.0};
      double quiet{0.0};
      bool live{false};
    };
    /** One leaf: where it is, where its solution and limiter status are stored, and its quiet time. */
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
    /** The score of a cell. complete is false if a neighbour was missing or the score was cut short. */
    struct ResolutionEvidence {
      double value{0.0};
      bool complete{true};
    };
    /** The score of a cell, computed once per step. state is 0 (empty), 1 (a thread computes it) or 2 (done). */
    struct DetailCache {
      std::atomic<int> state{0};
      ResolutionEvidence evidence{};
    };
    /** One face neighbour of a cell: its slot, the axis and side of the shared face, and the index of the
     * score of that face in _acceptedFaces. */
    struct NativeFace {
      int slot{-1}, normal{-1}, side{-1}, cache{-1};
    };
    /** The face neighbours of a cell. complete is false if one of them is missing. */
    struct NativeStencil {
      std::vector<NativeFace> faces;
      bool complete{false};
    };
    /** The score of the face between two leaves. Both cells use it, so it is computed once per step. */
    struct AcceptedFaceCache {
      std::atomic<int> state{0};
      int left{-1}, right{-1}, normal{-1};
      double value{0.0};
    };
    /** The slots of the children of one parent. complete is true if all of them are leaves. */
    struct Family {
      std::array<int, peano4::ChildrenPerCell> slots{};
      bool complete{false};
    };
    /** The proposed parent of a family and the checks that it has passed. */
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
    /** A leaf or one of its ancestors in legacy mode. An ancestor counts its own quiet time while all cells
     * below it are quiet. A parent then has a quiet time when its children coarsen. */
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
    /** Families with a child that could coarsen. assess() checks only these. */
    std::unique_ptr<std::atomic<bool>[]> _familyRequests;
    mutable std::atomic<std::size_t> _unboundVotes{0};
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
    bool _stencilsBound{false};
    bool _topologyChanged{true};
    bool _initialNeighbours{false};
    bool _assessed{false};
    bool _assessCoarsening{true};
    bool _warnedAboutStartTime{false};
    bool _warnedAboutUnboundVotes{false};
    /** What a cell decides on its own: it keeps its size, it refines, or it could coarsen if its siblings agree. */
    enum class OwnEvidence { Keep, Refine, Siblings };
    [[nodiscard]] OwnEvidence ownEvidence(const Cell& cell, const Coordinates& x, const Coordinates& h) const;

    // The private functions are described at their definitions in AMRCore.cpph.
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
    void bindTopology(const peano4::CellConnectivity& mesh);
    [[nodiscard]] double acceptedFaceDetail(int face) const;
    [[nodiscard]] double faceDetail(
      const AcceptedCellView<Precision>& own, const AcceptedCellView<Precision>& other,
      const Key& ownKey, const Key& otherKey, int normal, int side
    ) const;
    /** Sets up the grid geometry, sibling groups and transfer bounds for CPU or device assessment.
     * Returns true if the step starts at a finite, nonnegative time. */
    [[nodiscard]] bool prepareStep(const peano4::CellConnectivity& mesh, const exahype2::TimeStepping& clock,
                                   const double* nodes, const double* restriction, const double* prolongation);
    /** Adds the checks that one family passed to the statistics. */
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
