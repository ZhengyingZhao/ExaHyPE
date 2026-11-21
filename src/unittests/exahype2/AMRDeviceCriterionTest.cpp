// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "exahype2/amr/Criterion.h"
#include "exahype2/amr/DeviceCriterion.h"
#include "exahype2/dg/Basis.h"
#include "peano4/Spacetree.h"
#include "tarch/mpi/Rank.h"
#include "tarch/testing/UnitTest.h"
#include "unittests/peano4/TestGrids.h"

namespace {
  using Command     = exahype2::RefinementCommand;
  using Coordinates = tarch::la::Vector<DIMENSIONS, double>;
  using Control     = peano4::RefinementRequest::RefinementControl;

  TARGET_DECLARE
  template <bool FV, bool Prediction = true>
  struct CriterionConfig {
    static constexpr bool   IsFiniteVolume         = FV;
    static constexpr bool   UsePredictionCriterion = Prediction;
    static constexpr double TransferTolerance      = 0.01;
    static constexpr double ResolutionTolerance    = 0.01;
    static GPU_CALLABLE_METHOD double stateScale(int) { return 1.0; }
  };

  struct QuietDGConfig : CriterionConfig<false, false> {
    static constexpr double QuietTime     = 0.25;
    static constexpr double ReferenceTime = 1.0;
  };

  struct LinearDGConfig : CriterionConfig<false> {
    static GPU_CALLABLE_METHOD double quadratureNode(int node) {
      return node == 0 ? 0.21132486540518713 : 0.78867513459481287;
    }
    // For a linear polynomial, the Bernstein coefficients are the values at the two endpoints.
    // Each weight gives one Gauss node's contribution to an endpoint value.
    static GPU_CALLABLE_METHOD exahype2::amr::detail::transfer_bound::Interval transferWeight(int node, int mode) {
      namespace bound  = exahype2::amr::detail::transfer_bound;
      const double own = quadratureNode(node), other = quadratureNode(1 - node);
      return exahype2::amr::detail::retained_parent::quotient(
        bound::sub({double(mode), double(mode)}, {other, other}),
        bound::sub({own, own}, {other, other})
      );
    }
    static GPU_CALLABLE_METHOD double transferWeightLower(int node, int mode) { return transferWeight(node, mode).lo; }
    static GPU_CALLABLE_METHOD double transferWeightUpper(int node, int mode) { return transferWeight(node, mode).hi; }
  };

  struct StrictTransferDGConfig : LinearDGConfig {
    static constexpr double TransferTolerance   = 1.0e-5;
    static constexpr double ResolutionTolerance = 1.0;
  };

  struct Callbacks {
    static constexpr bool HasInitialDataCheck = false;
    bool                  allowTransfer{true};
    double                speed{0.0};
    template <typename P>
    GPU_CALLABLE_METHOD double waveSpeed(const P*, const Coordinates&, const Coordinates&, double, double, int) const {
      return speed;
    }
    template <typename P>
    GPU_CALLABLE_METHOD bool admissible(const P* Q, int states, const Coordinates&, const Coordinates&) const {
      if (Q == nullptr or states <= 0) {
        return false;
      }
      for (int node = 0; node < states; ++node) {
        if (not std::isfinite(Q[node]) or Q[node] <= 0.0) {
          return false;
        }
      }
      return true;
    }
    template <typename P>
    GPU_CALLABLE_METHOD bool transferAdmissible(const P*, const P* const*) const {
      return allowTransfer;
    }
    template <typename P>
    GPU_CALLABLE_METHOD bool initialDataPasses(const P*, const Coordinates&, const Coordinates&) const {
      return true;
    }
  };

  struct InitialCallbacks : Callbacks {
    static constexpr bool HasInitialDataCheck = true;
    bool                  acceptsSamples{true};
    GPU_CALLABLE_METHOD bool initialDataPasses(const double*, const Coordinates&, const Coordinates&) const {
      return acceptsSamples;
    }
  };

  struct ProposedParentConflict {
    int* results{nullptr};
    GPU_CALLABLE_METHOD void operator()(int entry) const {
      using Config   = CriterionConfig<true>;
      using Accepted = exahype2::amr::NativeAcceptedSolutionAdapter<1, 1, Config, double>;
      using Transfer = exahype2::amr::NativeTransferPolicy<1, 1, Config, double>;
      using View     = exahype2::amr::device_detail::Assessment<
            1,
            1,
            Config,
            double,
            Accepted,
            Transfer,
            exahype2::amr::NoLimiterEvidence,
            exahype2::amr::NativeAssessmentPolicy,
            Callbacks
          >;
      using Data                                             = typename View::Data;
      constexpr int                                 Siblings = peano4::ChildrenPerCell;
      std::array<typename Data::Cell, 2 * Siblings> cells{};
      std::array<typename Data::Family, 2>          families{};
      std::array<typename Data::FamilyPlan, 2>      plans{};
      std::array<typename Data::Face, 1>            faces{};
      std::array<typename Data::FamilyResult, 2>    familyResults{};
      familyResults[0].passed = true;
      familyResults[1].passed = entry != 2;
      for (int family = 0; family < 2; ++family) {
        families[family].complete = true;
        plans[family].eligible    = family == 0 or entry != 1;
        plans[family].parent[0]   = 1.0 + family;
        plans[family].key[0]      = 1;
        plans[family].key[1]      = family;
        for (int child = 0; child < Siblings; ++child) {
          const int slot                = family * Siblings + child;
          families[family].slots[child] = slot;
          cells[slot].slot              = slot;
          cells[slot].family            = family;
          cells[slot].complete          = true;
        }
      }
      cells[0].faceCount = 1;
      faces[0]           = {0, Siblings, 0, 1, 0.0};
      View view{};
      view._cells         = {cells.data(), static_cast<int>(cells.size())};
      view._faces         = {faces.data(), static_cast<int>(faces.size())};
      view._families      = {families.data(), static_cast<int>(families.size())};
      view._plans         = {plans.data(), static_cast<int>(plans.size())};
      view._familyResults = {familyResults.data(), static_cast<int>(familyResults.size())};
      results[entry]      = view.parentPassesNeighbours(families[0], plans[0].parent.data(), plans[0].key, true);
    }
  };
  TARGET_DECLARE_END

  struct HostInitialCheck {
    InitialCallbacks callbacks;
    bool operator()(const double* Q, const Coordinates& x, const Coordinates& h) const {
      return callbacks.initialDataPasses(Q, x, h);
    }
  };

  struct HostWaveSpeed {
    Callbacks callbacks;
    template <typename P>
    double operator()(
      const P*           Q,
      const Coordinates& x,
      const Coordinates& h,
      double             time,
      double             dt,
      int                normal
    ) const {
      return callbacks.waveSpeed(Q, x, h, time, dt, normal);
    }
  };
  struct HostAdmissible {
    Callbacks callbacks;
    template <typename P>
    bool operator()(const P* Q, int states, const Coordinates& x, const Coordinates& h) const {
      return callbacks.admissible(Q, states, x, h);
    }
  };
  struct HostTransferAdmissible {
    Callbacks callbacks;
    template <typename P>
    bool operator()(const P* parent, const P* const* children) const {
      return callbacks.transferAdmissible(parent, children);
    }
  };

  enum class State { Constant, Rough, Negative, NonFinite };

  // A level jump puts fine cells next to coarse ones around the centre of the grid.
  const peano4::CellConnectivity& getGrid(bool levelJump = false) {
    const int depth = levelJump ? peano4::unittests::DepthWithCoarseCellsBesideTheCentreBlock : 1;
    peano4::unittests::buildRegularGrid(depth);
    if (levelJump) {
      peano4::unittests::refineGrid({peano4::unittests::getRefinementOfCentreCell(depth)});
    }
    return peano4::Spacetree::getInstance().getCellConnectivity();
  }

  /** The host and device criteria share a grid and clock and each has its own copy of the solution. */
  template <int N, typename Config, typename P = double>
  struct Comparison {
    using Pool                                             = tarch::accelerator::ChunkedMemoryPool<P>;
    static constexpr int                            Values = peano4::power(N, DIMENSIONS);
    exahype2::amr::Criterion<N, 1, Config, P>       host;
    exahype2::amr::DeviceCriterion<N, 1, Config, P> device;
    Pool                                            hostPool{Values, tarch::MemoryLocation::Heap};
    Pool                                            devicePool{Values, tarch::MemoryLocation::Device};
    exahype2::TimeStepping                          clock{
      exahype2::TimeStepping::Rule::Fixed,
      1.0,
      exahype2::TimeStepping::Stability::PhysicalEigenvalue
    };
    exahype2::RefinementRequests                          requests;
    const peano4::CellConnectivity&                       mesh;
    Callbacks                                             callbacks{};
    std::array<double, N>                                 nodes{};
    std::array<double, peano4::SubdivisionFactor * N * N> restriction{}, prolongation{};

    explicit Comparison(const peano4::CellConnectivity& grid) :
      mesh(grid) {
      exahype2::Limits limits;
      limits.maxAdmissiblePatchSize = 1.0;
      limits.minAdmissiblePatchSize = 0.001;
      clock.setLimits(limits);
      hostPool.reserve(mesh.getNumberOfSlots());
      devicePool.reserve(mesh.getNumberOfSlots());
      static_assert(Config::IsFiniteVolume or N == 1 or N == 2);
      if constexpr (not Config::IsFiniteVolume and N == 2) {
        nodes = {Config::quadratureNode(0), Config::quadratureNode(1)};
        constexpr std::array<double, 2> weights{0.5, 0.5};
        // The solver indexes these matrices by child, input node, then output node.
        restriction = exahype2::dg::transposePerChild<N>(
          exahype2::dg::getWeightedRestrictionMatrix1D<N>(nodes.data(), weights.data())
        );
        prolongation = exahype2::dg::transposePerChild<N>(exahype2::dg::getInterpolationMatrix1D<N>(nodes.data()));
      } else {
        // Degree-zero DG uses exact constant interpolation and equal restriction weights.
        nodes.fill(0.5);
        restriction.fill(1.0 / peano4::SubdivisionFactor);
        prolongation.fill(1.0);
      }
      populate(State::Constant);
    }

    void populate(State state) {
      for (int slot : mesh.getCells()) {
        for (int node = 0; node < Values; ++node) {
          hostPool.getChunk(slot)[node] = static_cast<P>(
            state == State::NonFinite    ? std::numeric_limits<P>::infinity()
              : state == State::Negative ? -1.0
              : state == State::Rough    ? 1.0 + 0.5 * (node % 2)
                                         : 1.0
          );
        }
      }
      if (devicePool.isHostAccessible()) {
        for (int slot : mesh.getCells()) {
          std::copy_n(hostPool.getChunk(slot), Values, devicePool.getChunk(slot));
        }
      } else {
        devicePool.upload(hostPool);
      }
    }
    void begin() {
      host.beginStep(
        mesh,
        std::as_const(hostPool).getView(),
        clock,
        nodes.data(),
        restriction.data(),
        prolongation.data(),
        HostWaveSpeed{callbacks},
        HostAdmissible{callbacks},
        HostTransferAdmissible{callbacks}
      );
      device.beginStep(mesh, clock, nodes.data(), restriction.data(), prolongation.data());
    }
    void assessSelected(tarch::Span<const int> selected, const auto& terms, bool castVotes = true) {
      requests.clear(mesh.getNumberOfSlots());
      device.assess(
        mesh,
        std::as_const(devicePool).getView(),
        exahype2::amr::NoStatusView{},
        {},
        clock,
        requests,
        selected,
        terms,
        castVotes
      );
    }
    void assess(bool castVotes = true) { assessSelected(mesh.getCells(), callbacks, castVotes); }
    void assessInitial(const InitialCallbacks& samples) {
      host.beginInitialisation(HostAdmissible{samples}, HostInitialCheck{samples});
      device.beginInitialisation();
      assessSelected(mesh.getCells(), samples);
    }
    std::vector<Command> hostCommands() {
      std::vector<Command> result(mesh.getNumberOfSlots(), Command::Keep);
      for (int slot : mesh.getCells()) {
        const auto   width        = mesh.getCellWidth(slot);
        const double maximumWidth = tarch::la::max(width);
        const auto   command =
          host.command(hostPool.getChunk(slot), mesh.getCellCentre(slot), width, clock.getTimeStamp());
        result[slot] = tarch::la::greater(maximumWidth, clock.getLimits().maxAdmissiblePatchSize)
          ? Command::Refine
          : exahype2::clipRefinementVote(
              command,
              maximumWidth,
              clock.getLimits().maxAdmissiblePatchSize,
              clock.getLimits().minAdmissiblePatchSize
            );
      }
      return result;
    }
    void finish() {
      host.finishStep();
      device.finishStep();
    }
  };

  template <typename Comparison>
  void validateParity(Comparison& comparison, Command expected) {
    const auto  host   = comparison.hostCommands();
    const auto& device = comparison.device.commands();
    for (int slot : comparison.mesh.getCells()) {
      VALIDATE_EQUALS(static_cast<int>(device[slot]), static_cast<int>(host[slot]), slot);
      VALIDATE_EQUALS(static_cast<int>(device[slot]), static_cast<int>(expected), slot);
    }
  }
} // namespace

UNIT_TEST("Device FV AMR coarsens complete constant siblings and preserves CPU votes") {
  // This criterion needs all neighbouring cells on the same MPI rank.
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("One complete sibling family with an admissible constant accepted solution") {
    Comparison<2, CriterionConfig<true>> comparison(getGrid());
    comparison.begin();
    comparison.clock.advanceTimeStamp(0.125);

    WHEN("The device assesses the completed time step") {
      comparison.assess();

      THEN("Every sibling votes erase on both processors") {
        validateParity(comparison, Command::Erase);
        comparison.finish();
        const auto stats = comparison.device.statistics();
        VALIDATE_EQUALS(stats.assessedLeaves, comparison.mesh.getCells().size);
        VALIDATE_EQUALS(stats.assessedFamilies, 1);
        VALIDATE_EQUALS(stats.eligibleFamilies, 1);
        VALIDATE_EQUALS(stats.quietReady, 1);
        VALIDATE_EQUALS(stats.eraseVotes, 1);
        VALIDATE_EQUALS(stats.conflictVetoes, 0);
        const auto requests = comparison.requests.getLocalRefinementRequests(comparison.mesh);
        peano4::RefinementRequest parent;
        parent.setRefinementControl(Control::Erase);
        VALIDATE_EQUALS(requests.size(), std::size_t{1});
        VALIDATE_EQUALS(requests.front(), parent);
        comparison.begin();
        const auto nextStep = comparison.device.statistics();
        VALIDATE_EQUALS(nextStep.totalFamilies, 1);
        VALIDATE_EQUALS(nextStep.completeFamilies, 1);
        VALIDATE_EQUALS(nextStep.assessedFamilies, 0);
        VALIDATE_EQUALS(nextStep.assessedLeaves, 0);
      }
    }
  }
}

UNIT_TEST("Device AMR reads the accepted state after stepping and applies physical admissibility") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("A criterion bound before the solver changes its initially constant state") {
    Comparison<2, CriterionConfig<true>> comparison(getGrid());
    comparison.begin();

    WHEN("The accepted FV state gains unresolved within patch variation") {
      comparison.populate(State::Rough);
      comparison.clock.advanceTimeStamp(0.125);
      comparison.assess();

      THEN("Every cell refines and its command matches the host criterion") {
        validateParity(comparison, Command::Refine);
      }
    }

    WHEN("The accepted constant state becomes physically inadmissible") {
      comparison.populate(State::Negative);
      comparison.clock.advanceTimeStamp(0.125);
      comparison.assess();

      THEN("Physical admissibility requests refinement on both processors") {
        validateParity(comparison, Command::Refine);
      }
    }

    WHEN("A rough state is assessed with voting paused and then for one selected cell") {
      comparison.populate(State::Rough);
      comparison.clock.advanceTimeStamp(0.125);
      comparison.assess(false);
      VALIDATE(comparison.requests.getLocalRefinementRequests(comparison.mesh).empty());
      // The second solver asks to merge the siblings while the first solver's vote is paused.
      comparison.requests.merge(Command::Erase, comparison.mesh.getCells());
      VALIDATE_EQUALS(comparison.requests.getLocalRefinementRequests(comparison.mesh).size(), std::size_t{1});
      comparison.finish();
      comparison.begin();
      comparison.clock.advanceTimeStamp(0.125);
      const auto cells = comparison.mesh.getCells();
      const int  slot  = cells[cells.size - 1];
      comparison.assessSelected({&slot, 1}, comparison.callbacks);

      THEN("Only the selected cell has a mesh refinement request") {
        const auto requests = comparison.requests.getLocalRefinementRequests(comparison.mesh);
        peano4::RefinementRequest expected;
        expected.setRefinementControl(Control::Refine);
        expected.level        = comparison.mesh.getCellLevel(slot);
        expected.latticeIndex = comparison.mesh.getCellLattice(slot);
        VALIDATE_EQUALS(requests.size(), std::size_t{1});
        VALIDATE_EQUALS(requests.front(), expected);
      }
    }
  }
}

UNIT_TEST("Device AMR respects transfer vetoes and stale time steps") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("A constant sibling family") {
    Comparison<2, CriterionConfig<true>> comparison(getGrid());
    comparison.callbacks.allowTransfer = false;
    comparison.begin();

    WHEN("The parent transfer callback vetoes a completed time step") {
      comparison.clock.advanceTimeStamp(0.125);
      comparison.assess();

      THEN("Every cell keeps its current size on both processors") {
        validateParity(comparison, Command::Keep);
        VALIDATE_EQUALS(comparison.device.statistics().transferPassed, 0);
      }
    }

    WHEN("No accepted time step has advanced since beginStep") {
      comparison.assess();

      THEN("Both processors withhold refinement and coarsening votes") { validateParity(comparison, Command::Keep); }
    }
  }
}

UNIT_TEST("Device DG AMR carries quiet history across completed time steps") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("Degree zero DG siblings with a quiet time requirement of one quarter") {
    Comparison<1, QuietDGConfig> comparison(getGrid());
    comparison.begin();
    comparison.clock.advanceTimeStamp(0.125);
    comparison.assess(false);

    WHEN("A first quiet step finishes without casting votes after regrid") {
      validateParity(comparison, Command::Keep);
      comparison.finish();
      comparison.begin();
      comparison.clock.advanceTimeStamp(0.125);
      comparison.assess();

      THEN("The second accepted quiet step permits identical erase votes") {
        validateParity(comparison, Command::Erase);
        comparison.device.setReferenceTime(2.0);
        const auto reset = comparison.device.statistics();
        VALIDATE_EQUALS(reset.totalFamilies, 0);
        VALIDATE_EQUALS(reset.assessedLeaves, 0);
      }
    }
  }
}

UNIT_TEST("Degree one device DG AMR certifies constant sibling transfers in double and float") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  const auto compare = []<typename Precision>(Precision) {
    GIVEN("A complete family with Gauss Legendre nodes and the solver's degree one transfer matrices") {
      Comparison<2, LinearDGConfig, Precision> comparison(getGrid());
      comparison.begin();
      comparison.clock.advanceTimeStamp(0.125);

      WHEN("Device restriction prolongation and retained parent polynomial proofs assess the constant state") {
        comparison.assess();

        THEN("Every sibling erases and the completed transfer agrees with the CPU criterion") {
          validateParity(comparison, Command::Erase);
          VALIDATE_EQUALS(comparison.device.statistics().transferPassed, 1);
          VALIDATE_EQUALS(comparison.host.statistics().transferPassed, 1);
        }
      }
    }
  };
  compare(double{});
  compare(float{});
}

UNIT_TEST("Degree one device DG AMR refines rough accepted profiles in double and float") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  const auto compare = []<typename Precision>(Precision) {
    GIVEN("Degree one DG siblings bound before their accepted nodal states change") {
      Comparison<2, LinearDGConfig, Precision> comparison(getGrid());
      comparison.begin();
      comparison.populate(State::Rough);
      comparison.clock.advanceTimeStamp(0.125);

      WHEN("The accepted polynomials have unresolved jumps across patch boundaries") {
        comparison.assess();

        THEN("Every cell refines on both processors") { validateParity(comparison, Command::Refine); }
      }
    }
  };
  compare(double{});
  compare(float{});
}

UNIT_TEST("Degree one device DG AMR vetoes lossy transfers within the resolution budget in double and float") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  const auto compare = []<typename Precision>(Precision) {
    GIVEN("Resolved degree one sibling polynomials with a stricter fine to parent transfer budget") {
      Comparison<2, StrictTransferDGConfig, Precision> comparison(getGrid());
      comparison.populate(State::Rough);
      comparison.begin();
      comparison.clock.advanceTimeStamp(0.125);

      WHEN("Restriction and prolongation cannot preserve the accepted fine profiles within the transfer budget") {
        comparison.assess();

        THEN("Neither processor coarsens despite having no refinement evidence") {
          validateParity(comparison, Command::Keep);
          VALIDATE_EQUALS(comparison.device.statistics().refineEvidence, 0);
          VALIDATE_EQUALS(comparison.device.statistics().transferPassed, 0);
          VALIDATE_EQUALS(comparison.host.statistics().transferPassed, 0);
        }
      }
    }
  };
  compare(double{});
  compare(float{});
}

UNIT_TEST("Device AMR retains complete sibling and neighbour checks across a coarse fine grid") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("A constant accepted FV state on a grid with coarse fine interfaces") {
    Comparison<2, CriterionConfig<true>> comparison(getGrid(true));
    comparison.begin();
    comparison.clock.advanceTimeStamp(0.125);

    WHEN("The device evaluates accepted neighbours and proposed parent compatibility") {
      comparison.assess();
      const auto  host   = comparison.hostCommands();
      const auto& device = comparison.device.commands();
      int         erase = 0, keep = 0;

      THEN("Votes match the host while incomplete sibling families keep their cells") {
        VALIDATE(comparison.mesh.getLevelJumps(peano4::Adjacency::Local).size > 0);
        for (int slot : comparison.mesh.getCells()) {
          VALIDATE_EQUALS(static_cast<int>(device[slot]), static_cast<int>(host[slot]), slot);
          erase += device[slot] == Command::Erase;
          keep += device[slot] == Command::Keep;
        }
        VALIDATE(erase > 0);
        VALIDATE(keep > 0);
      }
    }
  }
}

UNIT_TEST("Device AMR initial sampling works before beginStep and preserves finite and physical checks") {
  if (tarch::mpi::Rank::getInstance().getNumberOfRanks() != 1) {
    return;
  }
  GIVEN("A criterion in initialisation with typed device sampling callbacks and no time step binding") {
    Comparison<2, CriterionConfig<true>> comparison(getGrid());
    InitialCallbacks                     callbacks;

    WHEN("Independent initial data samples reject a constant representation") {
      callbacks.acceptsSamples = false;
      comparison.assessInitial(callbacks);
      THEN("Both processors refine despite a constant stored state") { validateParity(comparison, Command::Refine); }
    }

    WHEN("Independent samples accept a finite admissible constant state") {
      comparison.assessInitial(callbacks);
      THEN("Both processors keep the initial cells") { validateParity(comparison, Command::Keep); }
    }

    WHEN("Sampling accepts but the initial state is nonfinite or physically inadmissible") {
      for (const State state : {State::NonFinite, State::Negative}) {
        comparison.populate(state);
        comparison.assessInitial(callbacks);
        validateParity(comparison, Command::Refine);
      }
    }
  }
}

UNIT_TEST("Device AMR proposed parent compatibility reads immutable phase one eligibility") {
  GIVEN("Two individually eligible families whose constant trial parents disagree across their common face") {
    tarch::accelerator::Buffer<int> results(4, tarch::MemoryLocation::Device);

    WHEN("A kernel checks both eligible parents an ineligible neighbour and two final vote values") {
      tarch::accelerator::parallelFor(tarch::accelerator::Range(4), ProposedParentConflict{results.getData()});
      const auto passed = results.getValues();

      THEN(
        "Eligible conflicting parents veto coarsening regardless of final votes while ineligible parents are ignored"
      ) {
        VALIDATE_EQUALS(passed[0], 0);
        VALIDATE_EQUALS(passed[1], 1);
        VALIDATE_EQUALS(passed[2], 0);
        VALIDATE_EQUALS(passed[3], 0);
      }
    }
  }
}
