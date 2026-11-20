// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "DeviceAssessment.h"

#include "exahype2/RefinementRequests.h"
#include "exahype2/RefinementVotes.h"
#include "tarch/accelerator/Buffer.h"

#include <optional>

namespace exahype2::amr::device_detail {
  TARGET_DECLARE
  /** Handles one cell, face or sibling group in an AMR kernel. */
  template <int Stage, typename View, typename SolutionView, typename StatusView>
  struct Evaluate {
    View                               view;
    SolutionView                       solution;
    StatusView                         statuses;
    exahype2::RefinementCommand*       commands;
    typename View::Data::QuietHistory* quietHistory;
    double                             minSize;
    bool                               initialising;
    bool                               active;

    GPU_CALLABLE_METHOD void operator()(int entry) const {
      using Command = exahype2::RefinementCommand;
      if constexpr (Stage == 7) {
        commands[entry] = Command::Keep;
        if (active) view._details[entry] = {};
      } else if constexpr (Stage == 0) {
        auto& cell  = view._cells[view._mesh.getCells()[entry]];
        cell.state  = solution.getChunk(cell.slot);
        cell.status = statuses.getChunk(cell.slot);
      } else if constexpr (Stage == 1) {
        auto& face = view._faces[entry];
        if (face.other < 0 or face.other >= view._cells.size) {
          face.value = INFINITY;
          return;
        }
        const auto& own   = view._cells[face.own];
        const auto& other = view._cells[face.other];
        // A shared face score uses the cell on the left as its first argument.
        if (view.cacheAcceptedFaces() and face.side == 0) {
          face.value = view.faceDetail(view.accepted(other), view.accepted(own), other.key, own.key, face.normal, 1);
        } else {
          face.value =
            view.faceDetail(view.accepted(own), view.accepted(other), own.key, other.key, face.normal, face.side);
        }
      } else if constexpr (Stage == 2) {
        const auto& cell         = view._cells[view._mesh.getCells()[entry]];
        view._details[cell.slot] = view.computeCellDetail(cell);
      } else if constexpr (Stage == 3) {
        auto& cell         = view._cells[view._mesh.getCells()[entry]];
        cell.acceptedQuiet = view.computeQuietProgress(cell, view._time, view._dt);
      } else if constexpr (Stage == 4) {
        view.prepareFamily(entry);
      } else if constexpr (Stage == 5) {
        const auto& plan = view._plans[entry];
        const bool  passed =
          plan.eligible and view.parentPassesNeighbours(view._families[entry], plan.parent.data(), plan.key, true);
        view._familyResults[entry] = {plan.sizeVetoed, plan.transferSafe, plan.neighboursSafe, plan.eligible, passed};
      } else if constexpr (Stage == 6) {
        const int    slot    = view._mesh.getCells()[entry];
        Command      command = Command::Keep;
        const double width   = tarch::la::max(view._mesh.getCellWidth(slot));
        if (tarch::la::greater(width, view._maxPatchSize)) {
          command = Command::Refine;
        } else if (tarch::la::greater(width, minSize) or peano4::SubdivisionFactor * width <= view._maxPatchSize) {
          command = initialising ? view.initialCommand(
                                     solution.getChunk(slot),
                                     view._mesh.getCellCentre(slot),
                                     view._mesh.getCellWidth(slot)
                                   )
            : active ? view.command(view._cells[slot])
                     : Command::Keep;
          command = exahype2::clipRefinementVote(command, width, view._maxPatchSize, minSize);
        }
        commands[slot] = command;
        if (active and not View::predictionMode()) {
          const auto& cell  = view._cells[slot];
          quietHistory[entry] = {cell.acceptedQuiet, cell.crossing, cell.moving};
        }
      }
    }
  };
  TARGET_DECLARE_END

} // namespace exahype2::amr::device_detail

namespace exahype2::amr {
  /** Device pointers to the DG nodes and transfer matrices. The tables must stay valid during assessment. */
  struct DeviceDGOperators {
    const double* nodes;
    const double* restriction;
    const double* prolongation;
  };

  /**
   * Uses the solver's accepted solution on the configured accelerator to decide which leaves
   * need refinement and which groups of siblings can be replaced by their parent.
   */
  template <
    int NodesPerAxis,
    int NumberOfData,
    typename Config,
    typename Precision  = double,
    typename Accepted   = NativeAcceptedSolutionAdapter<NodesPerAxis, NumberOfData, Config, Precision>,
    typename Transfer   = NativeTransferPolicy<NodesPerAxis, NumberOfData, Config, Precision>,
    typename Evidence   = NoLimiterEvidence,
    typename Assessment = DefaultAssessmentPolicy<Accepted>
  >
  class DeviceCriterion final {
    using Core = AMRCore<NodesPerAxis, NumberOfData, Config, Precision, Accepted, Transfer, Evidence, Assessment>;
    using Data = device_detail::Storage<NodesPerAxis, NumberOfData, Precision>;
    using Geometry = std::conditional_t<
      Transfer::IsFiniteVolume,
      detail::retained_parent::NoGeometry,
      detail::retained_parent::Geometry<NodesPerAxis, peano4::SubdivisionFactor>
    >;
    template <typename Value>
    using Buffer                     = tarch::accelerator::Buffer<Value>;
    static constexpr auto Device     = tarch::MemoryLocation::Device;
    static constexpr bool Prediction = Core::predictionMode();

  public:
    using Statistics     = typename Core::Statistics;
    using NativeSubcells = typename Core::NativeSubcells;
    using Coordinates    = typename Core::Coordinates;

    /** Starts a new assessment of which cells need refinement for the initial solution. */
    void beginInitialisation() {
      reset();
      _core.beginInitialisation();
    }

    /** Sets up the mesh and transfer tables for this time step. assess() uses the solution the solver accepts. */
    void beginStep(
      const peano4::CellConnectivity& mesh,
      const exahype2::TimeStepping&   clock,
      const double*                   nodes,
      const double*                   restriction,
      const double*                   prolongation,
      std::optional<DeviceDGOperators> deviceOperators = std::nullopt
    ) {
      _assessed = false;
      const bool validStartTime = _core.prepareStep(mesh, clock, nodes, restriction, prolongation);
      if constexpr (not Transfer::IsFiniteVolume) {
        if (restriction == nullptr or prolongation == nullptr) {
          throw std::invalid_argument("GPU AMR requires complete DG restriction and prolongation tables.");
        }
        if (deviceOperators) {
          if (not deviceOperators->nodes or not deviceOperators->restriction or not deviceOperators->prolongation) {
            throw std::invalid_argument("GPU AMR requires all three device DG operator tables.");
          }
          _deviceOperators = *deviceOperators;
        } else {
          constexpr int MatrixEntries = peano4::SubdivisionFactor * NodesPerAxis * NodesPerAxis;
          std::vector<double> tables(nodes, nodes + NodesPerAxis);
          tables.insert(tables.end(), restriction, restriction + MatrixEntries);
          tables.insert(tables.end(), prolongation, prolongation + MatrixEntries);
          upload(_operators, tables);
          const double* data = _operators.getData();
          _deviceOperators = {data, data + NodesPerAxis, data + NodesPerAxis + MatrixEntries};
        }
        ensure(_geometry, 1);
        _geometry.copyFromHost(&_core._retainedParentGeometry, 1);
      }
      _core._active = validStartTime;
    }

    /** Decides which leaves should split, keep their size or merge with their siblings.
     * Only selected leaves vote on the next mesh. In legacy mode the criterion still tracks
     * how long each cell has stayed quiet when castVotes is false. */
    template <typename SolutionView, typename StatusView, typename Callbacks>
    void assess(
      const peano4::CellConnectivity& mesh,
      SolutionView                    solution,
      StatusView                      statuses,
      NativeSubcells                  subcells,
      const exahype2::TimeStepping&   clock,
      exahype2::RefinementRequests&   requests,
      tarch::Span<const int>          selected,
      Callbacks                       callbacks,
      bool                            castVotes = true
    ) {
      static_assert(
        std::is_trivially_copyable_v<Callbacks>,
        "GPU AMR callbacks and RuntimeParameters must be trivially copyable device data."
      );
      using View = device_detail::
        Assessment<NodesPerAxis, NumberOfData, Config, Precision, Accepted, Transfer, Evidence, Assessment, Callbacks>;
      const bool initialising = _core._initialising;
      const bool active       = not initialising and _core.fresh(clock.getTimeStamp());
      const int  slots        = mesh.getNumberOfSlots();
      const auto leaves       = mesh.getCells();
      if (active) {
        prepareTopology(mesh);
      }
      ensure(_commands, slots);
      if constexpr (not Prediction) {
        if (active) {
          ensure(_quietHistory, leaves.size);
        }
      }
      View view{};
      view._cells         = span(_cells);
      view._faces         = span(_faces);
      view._families      = {_families.getData(), static_cast<int>(_families.getSize())};
      view._details       = span(_details);
      view._plans         = span(_plans);
      view._familyResults = span(_familyResults);
      view._subcells      = subcells;
      view._mesh          = mesh.onDevice();
      if constexpr (not Transfer::IsFiniteVolume) {
        if (active) {
          view._nodes                  = _deviceOperators.nodes;
          view._restriction            = _deviceOperators.restriction;
          view._prolongation           = _deviceOperators.prolongation;
          view._retainedParentGeometry = _geometry.getData();
        }
      }
      view._referenceTime     = _core._referenceTime;
      view._restrictionBound  = _core._restrictionBound;
      view._prolongationBound = _core._prolongationBound;
      view._time              = clock.getTimeStamp();
      view._dt                = clock.getTimeStepSizeOfLastStep();
      view._maxPatchSize      = clock.getLimits().maxAdmissiblePatchSize;
      view._callbacks         = callbacks;
      const auto run = [&]<int Stage>(int count) {
        tarch::accelerator::parallelFor(
          tarch::accelerator::Range(count),
          device_detail::Evaluate<Stage, View, SolutionView, StatusView>{
            view,
            solution,
            statuses,
            _commands.getData(),
            _quietHistory.getData(),
            clock.getLimits().minAdmissiblePatchSize,
            initialising,
            active
          }
        );
      };
      run.template operator()<7>(slots);
      if (active) {
        run.template operator()<0>(leaves.size);
        if constexpr (Assessment::CheckNeighbours) {
          run.template operator()<1>(static_cast<int>(_faces.getSize()));
        }
        run.template operator()<2>(leaves.size);
        if constexpr (not Prediction) {
          run.template operator()<3>(leaves.size);
        }
        run.template operator()<4>(static_cast<int>(_families.getSize()));
        run.template operator()<5>(static_cast<int>(_families.getSize()));
      }
      run.template operator()<6>(leaves.size);
      _commandsOnHost = _commands.getValues();
      if (active) {
        if constexpr (not Prediction) {
          const auto history = _quietHistory.getValues();
          for (int i = 0; i < leaves.size; ++i) {
            auto& cell         = _core._cells[leaves[i]];
            cell.acceptedQuiet = history[i].quiet;
            cell.crossing      = history[i].crossing;
            cell.moving        = history[i].moving;
          }
        }
        _assessed = true;
      }
      if (castVotes) {
        requests.merge(_commandsOnHost, selected);
      }
    }

    /** Looks up the vote for a cell by its centre, width and time. Returns Keep if no current vote exists. */
    [[nodiscard]] exahype2::RefinementCommand command(
      const Precision*,
      const Coordinates& x,
      const Coordinates& h,
      double             time
    ) const {
      if (not _assessed or not _core.fresh(time)) {
        return exahype2::RefinementCommand::Keep;
      }
      const auto found = _core._lookup.find(Core::geometry(x, h));
      return found == _core._lookup.end() ? exahype2::RefinementCommand::Keep : _commandsOnHost[found->second];
    }
    /** The most recent votes, indexed by mesh slot. */
    [[nodiscard]] const std::vector<exahype2::RefinementCommand>& commands() const { return _commandsOnHost; }
    /** Reports how many leaves exceed the resolution tolerance and how many sibling groups pass each
     * coarsening check. Before assessment, only the mesh's sibling groups are counted. */
    [[nodiscard]] Statistics statistics() const {
      Statistics result;
      result.totalFamilies = _core._families.size();
      for (const auto& family : _core._families) result.completeFamilies += family.complete;
      if (not _assessed or _core._mesh == nullptr) return result;

      for (const auto& family : _familyResults.getValues()) {
        Core::countAssessedFamily(result, family);
        result.eraseVotes += family.passed;
        result.conflictVetoes += family.eligible and not family.passed;
      }
      if constexpr (Assessment::CheckNeighbours) {
        const auto details = _details.getValues();
        result.assessedLeaves = _core._leaves.size();
        for (int slot : _core._leaves) {
          result.refineEvidence += details[slot].value > amr_detail::resolutionTolerance<Config>();
        }
      }
      return result;
    }
    void setReferenceTime(double time) { _core.setReferenceTime(time); }

    /** Ends this assessment. In legacy mode, a completed step carries quiet time forward for leaves and parents. */
    void finishStep() {
      if (not _core._active) {
        return;
      }
      if constexpr (not Prediction) {
        if (_assessed and _core._clock != nullptr and _core.fresh(_core._clock->getTimeStamp())) {
          const double time = _core._clock->getTimeStamp();
          for (int slot : _core._leaves) {
            auto& cell          = _core._cells[slot];
            cell.history->time  = time;
            cell.history->quiet = std::max(0., cell.acceptedQuiet);
          }
          _core.finishQuietRegions(time, _core._clock->getTimeStepSizeOfLastStep());
          if (_core._historyNeedsPruning) {
            for (auto entry = _core._history.begin(); entry != _core._history.end();) {
              if (not entry->second.live) {
                entry = _core._history.erase(entry);
              } else {
                ++entry;
              }
            }
            _core._historyNeedsPruning = false;
          }
        } else {
          _core._history.clear();
          _core._topologyCached      = false;
          _core._historyNeedsPruning = false;
        }
      }
      _core._active = false;
    }
    void reset() {
      _core.reset();
      _assessed = false;
      _cells.release();
      _faces.release();
      _families.release();
      _details.release();
      _plans.release();
      _commands.release();
      _quietHistory.release();
      _familyResults.release();
      _operators.release();
      _deviceOperators = {};
      _geometry.release();
      _commandsOnHost.clear();
    }

  private:
    Core                                      _core;
    bool                                      _assessed{false};
    Buffer<typename Data::Cell>               _cells{Device};
    Buffer<typename Data::Face>               _faces{Device};
    Buffer<typename Data::Family>             _families{Device};
    Buffer<typename Data::ResolutionEvidence> _details{Device};
    Buffer<typename Data::FamilyPlan>         _plans{Device};
    Buffer<typename Data::QuietHistory>       _quietHistory{Device};
    Buffer<typename Data::FamilyResult>       _familyResults{Device};
    Buffer<exahype2::RefinementCommand>       _commands{Device};
    Buffer<double>                            _operators{Device};
    DeviceDGOperators                         _deviceOperators{};
    Buffer<Geometry>                          _geometry{Device};
    std::vector<exahype2::RefinementCommand>  _commandsOnHost;

    template <typename Value>
    static void ensure(Buffer<Value>& buffer, std::size_t size) {
      if (buffer.getSize() != size) {
        buffer.allocate(size);
      }
    }
    template <typename Value>
    static void upload(Buffer<Value>& buffer, const std::vector<Value>& values) {
      ensure(buffer, values.size());
      buffer.copyFromHost(values.data(), values.size());
    }
    template <typename Value>
    static tarch::Span<Value> span(Buffer<Value>& buffer) {
      return {buffer.getData(), static_cast<int>(buffer.getSize())};
    }
    /** Gives the device the face neighbours and sibling group for each leaf. */
    void prepareTopology(const peano4::CellConnectivity& mesh) {
      std::vector<typename Data::Cell> cells(_core._cells.size());
      std::vector<typename Data::Face> faces;
      for (int slot : _core._leaves) {
        const auto& source = _core._cells[slot];
        auto&       target = cells[slot];
        target.key         = source.key;
        target.slot        = slot;
        target.family      = source.family;
        target.centre      = source.centre;
        target.width       = source.width;
        if constexpr (not Prediction) {
          target.quiet = source.history->time == _core._startTime ? source.history->quiet : 0.0;
        }
        target.firstFace   = static_cast<int>(faces.size());
        if constexpr (Assessment::CheckNeighbours or Assessment::UseLocalConstancy) {
          target.complete = forEachNativeNeighbour(mesh, slot, [&](int neighbour, int normal, int side) {
            faces.push_back({slot, neighbour, normal, side, 0.});
          });
        } else {
          target.complete = true;
        }
        target.faceCount = static_cast<int>(faces.size()) - target.firstFace;
      }
      std::vector<typename Data::Family> families;
      for (const auto& family : _core._families) {
        families.push_back({family.slots, family.complete});
      }
      upload(_cells, cells);
      upload(_faces, faces);
      upload(_families, families);
      ensure(_details, cells.size());
      ensure(_plans, families.size());
      ensure(_familyResults, families.size());
    }
  };
} // namespace exahype2::amr
