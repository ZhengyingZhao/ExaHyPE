// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "exahype2/RefinementVotes.h"

#include "exahype2/Limits.h"
#include "exahype2/RefinementCommand.h"
#include "exahype2/RefinementRequests.h"
#include "peano4/CellConnectivity.h"
#include "peano4/Spacetree.h"
#include "tarch/CompilerSpecificSettings.h"
#include "tarch/accelerator/ChunkedMemoryPool.h"
#include "tarch/accelerator/MemoryLocation.h"
#include "tarch/accelerator/ParallelFor.h"
#include "tarch/accelerator/Range.h"
#include "tarch/accelerator/accelerator.h"
#include "tarch/la/Vector.h"
#include "tarch/testing/UnitTest.h"
#include "unittests/peano4/TestGrids.h"

#include <atomic>

namespace {
  constexpr double MaxAdmissibleSize = 1.0;
  constexpr double MinAdmissibleSize = 0.1;
  constexpr double TimeStamp         = 0.5;

  /**
   * Solver whose refinement criterion returns the command encoded in the first value of the state.
   * The runtime parameter refinesEveryCell replaces this command by Refine.
   */
  struct StubSolver final {
    struct RuntimeParameters final {
      bool refinesEveryCell{false};
    };

    static GPU_CALLABLE_METHOD exahype2::RefinementCommand refinementCriterion(
      const double* const                                           Q,
      [[maybe_unused]] const tarch::la::Vector<DIMENSIONS, double>& centre,
      [[maybe_unused]] const tarch::la::Vector<DIMENSIONS, double>& width,
      [[maybe_unused]] const double                                 timeStamp,
      const RuntimeParameters&                                      parameters
    ) {
      if (parameters.refinesEveryCell) {
        return exahype2::RefinementCommand::Refine;
      }
      return Q == nullptr ? exahype2::RefinementCommand::Keep
                          : static_cast<exahype2::RefinementCommand>(static_cast<std::uint8_t>(Q[0]));
    }
  };

  constexpr StubSolver::RuntimeParameters CommandOfThePatch{};
  constexpr StubSolver::RuntimeParameters RefinementOfEveryCell{.refinesEveryCell = true};

  /** Deliberately host-only: a GPU vote kernel must never instantiate this callback. */
  struct HostCriterionSolver final {
    using RuntimeParameters                                  = StubSolver::RuntimeParameters;
    static constexpr bool          RefinementCriterionOnHost = true;
    inline static std::atomic<int> calls{0};

    static exahype2::RefinementCommand refinementCriterion(
      const double*                                Q,
      const tarch::la::Vector<DIMENSIONS, double>& centre,
      const tarch::la::Vector<DIMENSIONS, double>& width,
      double                                       timeStamp,
      const RuntimeParameters&                     parameters
    ) {
      ++calls;
      return StubSolver::refinementCriterion(Q, centre, width, timeStamp, parameters);
    }
  };

  int getNumberOf(const exahype2::RefinementCommand command) { return static_cast<int>(command); }

  constexpr int Depth = 1;

  using Pool = tarch::accelerator::ChunkedMemoryPool<double>;

  const peano4::CellConnectivity& buildGrid() {
    peano4::unittests::buildRegularGrid(Depth);
    return peano4::Spacetree::getInstance().getCellConnectivity();
  }

  TARGET_DECLARE
  struct WriteCommandIntoEveryCell final {
    peano4::CellConnectivity cellConnectivity{};
    Pool::View               solution{};
    double                   code{0.0};

    INLINE_METHOD GPU_CALLABLE_METHOD void operator()(const int entry) const {
      solution.getChunk(cellConnectivity.getCells()[entry])[0] = code;
    }
  };
  TARGET_DECLARE_END

  void writeCommandIntoEveryCell(
    Pool&                             pool,
    const peano4::CellConnectivity&   cellConnectivity,
    const exahype2::RefinementCommand command
  ) {
    tarch::accelerator::parallelFor(
      tarch::accelerator::Range(cellConnectivity.getCells().size),
      WriteCommandIntoEveryCell{
        cellConnectivity.onDevice(),
        pool.getView(),
        static_cast<double>(static_cast<std::uint8_t>(command))
      }
    );
  }

  /**
   * Returns limits that leave every vote on the regular grid of depth Depth to the criterion.
   */
  exahype2::Limits getLimitsAroundTheGrid() {
    exahype2::Limits limits;
    limits.minAdmissiblePatchSize = 0.0;
    limits.maxAdmissiblePatchSize = 1.0;
    return limits;
  }

  exahype2::RefinementCommand getVoteOf(const exahype2::RefinementCommand criterion, const double width) {
    const auto state = static_cast<double>(static_cast<std::uint8_t>(criterion));
    return exahype2::voteWithinAdmissibleSizes<StubSolver>(
      &state,
      tarch::la::Vector<DIMENSIONS, double>(0.5),
      tarch::la::Vector<DIMENSIONS, double>(width),
      MaxAdmissibleSize,
      MinAdmissibleSize,
      TimeStamp,
      CommandOfThePatch
    );
  }
} // namespace

UNIT_TEST("A cell wider than the largest admissible size refines whatever the criterion says") {
  GIVEN("A cell twice as wide as the largest admissible size") {
    constexpr double Width = 2.0 * MaxAdmissibleSize;

    WHEN("The criterion says keep or erase") {
      const exahype2::RefinementCommand voteOfKeep  = getVoteOf(exahype2::RefinementCommand::Keep, Width);
      const exahype2::RefinementCommand voteOfErase = getVoteOf(exahype2::RefinementCommand::Erase, Width);

      THEN("Both votes refine") {
        VALIDATE_EQUALS(getNumberOf(voteOfKeep), getNumberOf(exahype2::RefinementCommand::Refine));
        VALIDATE_EQUALS(getNumberOf(voteOfErase), getNumberOf(exahype2::RefinementCommand::Refine));
      }
    }
  }
}

UNIT_TEST("A cell between the two admissible sizes takes the command of the criterion") {
  GIVEN("A cell whose width is the largest admissible size over the subdivision factor") {
    constexpr double Width = MaxAdmissibleSize / peano4::SubdivisionFactor;

    WHEN("The criterion says refine, keep or erase") {
      const exahype2::RefinementCommand voteOfRefine = getVoteOf(exahype2::RefinementCommand::Refine, Width);
      const exahype2::RefinementCommand voteOfKeep   = getVoteOf(exahype2::RefinementCommand::Keep, Width);
      const exahype2::RefinementCommand voteOfErase  = getVoteOf(exahype2::RefinementCommand::Erase, Width);

      THEN("Every vote is the command of the criterion") {
        VALIDATE_EQUALS(getNumberOf(voteOfRefine), getNumberOf(exahype2::RefinementCommand::Refine));
        VALIDATE_EQUALS(getNumberOf(voteOfKeep), getNumberOf(exahype2::RefinementCommand::Keep));
        VALIDATE_EQUALS(getNumberOf(voteOfErase), getNumberOf(exahype2::RefinementCommand::Erase));
      }
    }
  }
}

UNIT_TEST("A refine vote turns into keep when the children would fall below the smallest admissible size") {
  GIVEN("A cell of the smallest admissible size") {
    constexpr double Width = MinAdmissibleSize;

    WHEN("The criterion says refine") {
      const exahype2::RefinementCommand vote = getVoteOf(exahype2::RefinementCommand::Refine, Width);

      THEN("The vote keeps the cell") {
        VALIDATE_EQUALS(getNumberOf(vote), getNumberOf(exahype2::RefinementCommand::Keep));
      }
    }
  }
}

UNIT_TEST("An erase vote turns into keep when the parent would pass the largest admissible size") {
  GIVEN("A cell of the largest admissible size") {
    constexpr double Width = MaxAdmissibleSize;

    WHEN("The criterion says erase") {
      const exahype2::RefinementCommand vote = getVoteOf(exahype2::RefinementCommand::Erase, Width);

      THEN("The vote keeps the cell") {
        VALIDATE_EQUALS(getNumberOf(vote), getNumberOf(exahype2::RefinementCommand::Keep));
      }
    }
  }
}

UNIT_TEST("The votes pass a refine vote of every cell into the requests, drop a keep vote and free in release()") {
  GIVEN("A regular grid over every rank, a pool of one value per cell and the votes and requests of a solver") {
    const auto& cellConnectivity = buildGrid();
    const int   numberOfSlots    = cellConnectivity.getNumberOfSlots();
    Pool        pool(1, tarch::MemoryLocation::Device);
    pool.reserve(numberOfSlots);
    exahype2::RefinementVotes    votes;
    exahype2::RefinementRequests requests;

    WHEN("Every patch says refine and the solver votes on every cell") {
      writeCommandIntoEveryCell(pool, cellConnectivity, exahype2::RefinementCommand::Refine);
      requests.clear(numberOfSlots);
      votes.voteOnEveryCell<StubSolver>(
        cellConnectivity,
        std::as_const(pool).getView(),
        getLimitsAroundTheGrid(),
        TimeStamp,
        requests,
        CommandOfThePatch
      );
      const auto        refinementRequests = requests.getRefinementRequests(cellConnectivity);
      const std::size_t allocatedBytes     = votes.getAllocatedBytes().getTotal();

      THEN("Refinement requests arise and the votes occupy at least one byte per slot") {
        VALIDATE(not refinementRequests.empty());
        VALIDATE(allocatedBytes >= static_cast<std::size_t>(numberOfSlots));
      }

      AND_WHEN("Every patch says keep and the solver votes on every cell again") {
        writeCommandIntoEveryCell(pool, cellConnectivity, exahype2::RefinementCommand::Keep);
        requests.clear(numberOfSlots);
        votes.voteOnEveryCell<StubSolver>(
          cellConnectivity,
          std::as_const(pool).getView(),
          getLimitsAroundTheGrid(),
          TimeStamp,
          requests,
          CommandOfThePatch
        );
        const auto refinementRequestsOfTheKeep = requests.getRefinementRequests(cellConnectivity);

        THEN("The refinement requests stay empty") { VALIDATE(refinementRequestsOfTheKeep.empty()); }

        AND_WHEN("The solver votes on every cell again with the runtime parameter that refines every cell") {
          requests.clear(numberOfSlots);
          votes.voteOnEveryCell<StubSolver>(
            cellConnectivity,
            std::as_const(pool).getView(),
            getLimitsAroundTheGrid(),
            TimeStamp,
            requests,
            RefinementOfEveryCell
          );
          const auto refinementRequestsOfTheParameter = requests.getRefinementRequests(cellConnectivity);

          THEN("Refinement requests arise") { VALIDATE(not refinementRequestsOfTheParameter.empty()); }
        }

        AND_WHEN("The votes release their commands") {
          votes.release();
          const std::size_t bytesAfterTheRelease = votes.getAllocatedBytes().getTotal();

          THEN("The votes count zero bytes") { VALIDATE_EQUALS(bytesAfterTheRelease, std::size_t{0}); }
        }
      }
    }
  }
}

UNIT_TEST("Without patches, every cell above the largest admissible size refines and every other cell keeps its size") {
  GIVEN("A regular grid over every rank, the votes and requests of a solver and the limits around the grid") {
    const auto&                  cellConnectivity = buildGrid();
    exahype2::RefinementVotes    votes;
    exahype2::RefinementRequests requests;
    exahype2::Limits             limits = getLimitsAroundTheGrid();

    WHEN("The solver votes on the grid construction with a largest admissible size below the cell width") {
      limits.maxAdmissiblePatchSize = 0.5 / peano4::SubdivisionFactor;
      requests.clear(cellConnectivity.getNumberOfSlots());
      votes.voteOnGridConstruction<StubSolver, double>(cellConnectivity, limits, requests, CommandOfThePatch);
      const auto refinementRequests = requests.getRefinementRequests(cellConnectivity);

      THEN("Refinement requests arise") { VALIDATE(not refinementRequests.empty()); }
    }

    WHEN("The solver votes on the grid construction with the largest admissible size one") {
      limits.maxAdmissiblePatchSize = 1.0;
      requests.clear(cellConnectivity.getNumberOfSlots());
      votes.voteOnGridConstruction<StubSolver, double>(cellConnectivity, limits, requests, CommandOfThePatch);
      const auto refinementRequests = requests.getRefinementRequests(cellConnectivity);

      THEN("The refinement requests stay empty") { VALIDATE(refinementRequests.empty()); }
    }
  }
}

UNIT_TEST("Only the listed cells vote whatever the patches of the other cells say") {
  GIVEN("A regular grid over every rank whose patches all say refine and the votes and requests of a solver") {
    const auto& cellConnectivity = buildGrid();
    const int   numberOfSlots    = cellConnectivity.getNumberOfSlots();
    Pool        pool(1, tarch::MemoryLocation::Device);
    pool.reserve(numberOfSlots);
    writeCommandIntoEveryCell(pool, cellConnectivity, exahype2::RefinementCommand::Refine);
    exahype2::RefinementVotes    votes;
    exahype2::RefinementRequests requests;

    WHEN("The solver votes on an empty list of cells") {
      const std::vector<int> listed;
      requests.clear(numberOfSlots);
      votes.voteOnCells<StubSolver>(
        cellConnectivity,
        listed,
        std::as_const(pool).getView(),
        getLimitsAroundTheGrid(),
        TimeStamp,
        requests,
        CommandOfThePatch
      );
      const auto refinementRequests = requests.getRefinementRequests(cellConnectivity);

      THEN("The refinement requests stay empty") { VALIDATE(refinementRequests.empty()); }
    }

    WHEN("The solver votes on a list of the first cell of every rank") {
      std::vector<int> listed;
      if (not cellConnectivity.getCells().isEmpty()) {
        listed.push_back(cellConnectivity.getCells()[0]);
      }
      requests.clear(numberOfSlots);
      votes.voteOnCells<StubSolver>(
        cellConnectivity,
        listed,
        std::as_const(pool).getView(),
        getLimitsAroundTheGrid(),
        TimeStamp,
        requests,
        CommandOfThePatch
      );
      const auto refinementRequests = requests.getRefinementRequests(cellConnectivity);

      THEN("Refinement requests arise") { VALIDATE(not refinementRequests.empty()); }
    }
  }
}

UNIT_TEST("Limits that fix the mesh skip the votes") {
  GIVEN("A regular grid over every rank, limits that admit its cell width alone and a refine vote in every cell") {
    const auto&      cellConnectivity = buildGrid();
    const int        numberOfSlots    = cellConnectivity.getNumberOfSlots();
    const double     width            = cellConnectivity.getWidthOfLevel(cellConnectivity.getMaximumLevel(), 0);
    exahype2::Limits fixingLimits;
    fixingLimits.minAdmissiblePatchSize = width;
    fixingLimits.maxAdmissiblePatchSize = width;
    Pool pool(1, tarch::MemoryLocation::Device);
    pool.reserve(numberOfSlots);
    writeCommandIntoEveryCell(pool, cellConnectivity, exahype2::RefinementCommand::Refine);

    WHEN("The solver votes on every cell under the fixing limits and then under limits around the grid") {
      exahype2::RefinementVotes    votes;
      exahype2::RefinementRequests requests;
      requests.clear(numberOfSlots);
      votes.voteOnEveryCell<StubSolver>(
        cellConnectivity,
        std::as_const(pool).getView(),
        fixingLimits,
        TimeStamp,
        requests,
        CommandOfThePatch
      );
      const auto fixedRequests = requests.getRefinementRequests(cellConnectivity);
      requests.clear(numberOfSlots);
      votes.voteOnEveryCell<StubSolver>(
        cellConnectivity,
        std::as_const(pool).getView(),
        getLimitsAroundTheGrid(),
        TimeStamp,
        requests,
        CommandOfThePatch
      );
      const auto openRequests = requests.getRefinementRequests(cellConnectivity);
      votes.release();

      THEN("The fixing limits leave the requests empty and the limits around the grid raise requests") {
        VALIDATE(fixedRequests.empty());
        VALIDATE(not openRequests.empty());
      }
    }
  }
}

UNIT_TEST("A host-only criterion reads the downloaded device solution and refreshes its votes") {
  const auto& cellConnectivity = buildGrid();
  const int   numberOfSlots    = cellConnectivity.getNumberOfSlots();
  Pool        deviceSolution(1, tarch::MemoryLocation::Device);
  Pool        hostSolution(1, tarch::MemoryLocation::Heap);
  deviceSolution.reserve(numberOfSlots);
  exahype2::RefinementVotes    votes;
  exahype2::RefinementRequests requests;

  writeCommandIntoEveryCell(deviceSolution, cellConnectivity, exahype2::RefinementCommand::Refine);
  const auto acceptedBefore  = std::as_const(deviceSolution).download(hostSolution).getView();
  HostCriterionSolver::calls = 0;
  requests.clear(numberOfSlots);
  votes.voteOnEveryCell<HostCriterionSolver>(
    cellConnectivity,
    acceptedBefore,
    getLimitsAroundTheGrid(),
    TimeStamp,
    requests,
    CommandOfThePatch
  );
  const auto refineRequests = requests.getRefinementRequests(cellConnectivity);
  VALIDATE(not refineRequests.empty());
  VALIDATE_EQUALS(HostCriterionSolver::calls.load(), cellConnectivity.getCells().size);

  writeCommandIntoEveryCell(deviceSolution, cellConnectivity, exahype2::RefinementCommand::Keep);
  const auto acceptedAfter = std::as_const(deviceSolution).download(hostSolution).getView();
  for (const int slot : cellConnectivity.getCells()) {
    VALIDATE(acceptedBefore.getChunk(slot) == acceptedAfter.getChunk(slot));
    VALIDATE_EQUALS(acceptedBefore.getChunk(slot)[0], static_cast<double>(exahype2::RefinementCommand::Keep));
  }
  requests.clear(numberOfSlots);
  votes.voteOnEveryCell<HostCriterionSolver>(
    cellConnectivity,
    acceptedAfter,
    getLimitsAroundTheGrid(),
    TimeStamp,
    requests,
    CommandOfThePatch
  );
  VALIDATE(requests.getRefinementRequests(cellConnectivity).empty());

  requests.clear(numberOfSlots);
  votes.voteOnEveryCell<HostCriterionSolver>(
    cellConnectivity,
    acceptedAfter,
    getLimitsAroundTheGrid(),
    TimeStamp,
    requests,
    RefinementOfEveryCell
  );
  VALIDATE(requests.getRefinementRequests(cellConnectivity) == refineRequests);

  votes.release();
  VALIDATE_EQUALS(votes.getAllocatedBytes().getTotal(), std::size_t{0});
}

UNIT_TEST("Host-only and device criteria cast the same grid-construction and listed-cell votes") {
  const auto&                  cellConnectivity = buildGrid();
  const int                    numberOfSlots    = cellConnectivity.getNumberOfSlots();
  exahype2::RefinementVotes    hostVotes, deviceVotes;
  exahype2::RefinementRequests hostRequests, deviceRequests;
  hostRequests.clear(numberOfSlots);
  deviceRequests.clear(numberOfSlots);
  HostCriterionSolver::calls = 0;
  hostVotes.voteOnGridConstruction<HostCriterionSolver, double>(
    cellConnectivity,
    getLimitsAroundTheGrid(),
    hostRequests,
    RefinementOfEveryCell
  );
  deviceVotes.voteOnGridConstruction<StubSolver, double>(
    cellConnectivity,
    getLimitsAroundTheGrid(),
    deviceRequests,
    RefinementOfEveryCell
  );
  VALIDATE(
    hostRequests.getRefinementRequests(cellConnectivity) == deviceRequests.getRefinementRequests(cellConnectivity)
  );
  VALIDATE_EQUALS(HostCriterionSolver::calls.load(), cellConnectivity.getCells().size);

  Pool deviceSolution(1, tarch::MemoryLocation::Device);
  Pool hostSolution(1, tarch::MemoryLocation::Heap);
  deviceSolution.reserve(numberOfSlots);
  writeCommandIntoEveryCell(deviceSolution, cellConnectivity, exahype2::RefinementCommand::Refine);
  const auto       downloaded = std::as_const(deviceSolution).download(hostSolution).getView();
  std::vector<int> listed;
  for (int entry = 0; entry < cellConnectivity.getCells().size; entry += 2) {
    listed.push_back(cellConnectivity.getCells()[entry]);
  }
  for (int selection = 0; selection < 2; ++selection) {
    hostRequests.clear(numberOfSlots);
    deviceRequests.clear(numberOfSlots);
    HostCriterionSolver::calls = 0;
    hostVotes.voteOnCells<HostCriterionSolver>(
      cellConnectivity,
      listed,
      downloaded,
      getLimitsAroundTheGrid(),
      TimeStamp,
      hostRequests,
      CommandOfThePatch
    );
    deviceVotes.voteOnCells<StubSolver>(
      cellConnectivity,
      listed,
      std::as_const(deviceSolution).getView(),
      getLimitsAroundTheGrid(),
      TimeStamp,
      deviceRequests,
      CommandOfThePatch
    );
    VALIDATE(
      hostRequests.getRefinementRequests(cellConnectivity) == deviceRequests.getRefinementRequests(cellConnectivity)
    );
    VALIDATE_EQUALS(HostCriterionSolver::calls.load(), static_cast<int>(listed.size()));
    listed.clear();
  }
}

UNIT_TEST("Host-only criteria obey mesh size limits without calling a fixed-mesh criterion") {
  const auto&  cellConnectivity = buildGrid();
  const int    numberOfSlots    = cellConnectivity.getNumberOfSlots();
  const double width            = cellConnectivity.getWidthOfLevel(cellConnectivity.getMaximumLevel(), 0);
  Pool         hostSolution(1, tarch::MemoryLocation::Heap);
  hostSolution.reserve(numberOfSlots);
  for (const int slot : cellConnectivity.getCells()) {
    hostSolution.getChunk(slot)[0] = static_cast<double>(exahype2::RefinementCommand::Refine);
  }
  exahype2::Limits limits;
  limits.minAdmissiblePatchSize = width;
  limits.maxAdmissiblePatchSize = width;
  exahype2::RefinementVotes    votes;
  exahype2::RefinementRequests requests;
  requests.clear(numberOfSlots);
  HostCriterionSolver::calls = 0;
  votes.voteOnEveryCell<HostCriterionSolver>(
    cellConnectivity,
    std::as_const(hostSolution).getView(),
    limits,
    TimeStamp,
    requests,
    RefinementOfEveryCell
  );
  VALIDATE(requests.getRefinementRequests(cellConnectivity).empty());
  VALIDATE_EQUALS(HostCriterionSolver::calls.load(), 0);

  // A listed vote still passes through the same clipping rules.
  std::vector<int> listed;
  for (const int slot : cellConnectivity.getCells()) {
    listed.push_back(slot);
  }
  requests.clear(numberOfSlots);
  votes.voteOnCells<HostCriterionSolver>(
    cellConnectivity,
    listed,
    std::as_const(hostSolution).getView(),
    limits,
    TimeStamp,
    requests,
    RefinementOfEveryCell
  );
  VALIDATE(requests.getRefinementRequests(cellConnectivity).empty());
}

UNIT_TEST("Host and device votes agree for refinement and erasure at either size guard") {
  const auto&  cellConnectivity = buildGrid();
  const int    numberOfSlots    = cellConnectivity.getNumberOfSlots();
  const double width            = cellConnectivity.getWidthOfLevel(cellConnectivity.getMaximumLevel(), 0);
  Pool         deviceSolution(1, tarch::MemoryLocation::Device);
  Pool         hostSolution(1, tarch::MemoryLocation::Heap);
  deviceSolution.reserve(numberOfSlots);
  exahype2::RefinementVotes    hostVotes, deviceVotes;
  exahype2::RefinementRequests hostRequests, deviceRequests;
  for (const auto command :
       {exahype2::RefinementCommand::Refine, exahype2::RefinementCommand::Keep, exahype2::RefinementCommand::Erase}) {
    writeCommandIntoEveryCell(deviceSolution, cellConnectivity, command);
    const auto downloaded = std::as_const(deviceSolution).download(hostSolution).getView();
    for (int guard = 0; guard < 4; ++guard) {
      auto limits = getLimitsAroundTheGrid();
      if ((guard & 1) != 0) {
        limits.minAdmissiblePatchSize = width;
      }
      if ((guard & 2) != 0) {
        limits.maxAdmissiblePatchSize = width;
      }
      hostRequests.clear(numberOfSlots);
      deviceRequests.clear(numberOfSlots);
      hostVotes.voteOnEveryCell<HostCriterionSolver>(
        cellConnectivity,
        downloaded,
        limits,
        TimeStamp,
        hostRequests,
        CommandOfThePatch
      );
      deviceVotes.voteOnEveryCell<StubSolver>(
        cellConnectivity,
        std::as_const(deviceSolution).getView(),
        limits,
        TimeStamp,
        deviceRequests,
        CommandOfThePatch
      );
      VALIDATE(
        hostRequests.getRefinementRequests(cellConnectivity) == deviceRequests.getRefinementRequests(cellConnectivity)
      );
    }
  }
}
