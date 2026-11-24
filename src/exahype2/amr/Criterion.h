// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "AMRCore.h"
#include <type_traits>

namespace exahype2::amr {
  namespace detail {
    /** The limiter adapter of an Accepted type, or void if it has none. */
    template<typename Accepted, typename = void>
    struct AcceptedLimiterLayout { using type = void; };
    template<typename Accepted>
    struct AcceptedLimiterLayout<Accepted, std::void_t<typename Accepted::LimiterAdapter>> {
      using type = typename Accepted::LimiterAdapter;
    };
  }

  /**
   * The AMR criterion that runs on the CPU, for FV and DG solvers and for ADER-DG with the subcell limiter.
   * A generated solver uses this class. AMRCore does the work and explains the decisions.
   *
   * In every time step the solver calls beginStep() before the step and, once the step is complete, assess(),
   * command() for every cell and finishStep(). The solution given to beginStep() and to command() has to be
   * in host memory. A solver that computes on a GPU copies its solution there after the step, before the votes.
   */
  template<int NodesPerAxis, int NumberOfData, typename Config, typename Precision = double,
           typename Accepted = NativeAcceptedSolutionAdapter<NodesPerAxis, NumberOfData, Config, Precision>,
           typename Transfer = NativeTransferPolicy<NodesPerAxis, NumberOfData, Config, Precision>,
           typename Evidence = NoLimiterEvidence,
           typename Assessment = DefaultAssessmentPolicy<Accepted>>
  class Criterion final {
  public:
    using AcceptedAdapter = Accepted;
    using TransferPolicy = Transfer;
    using StabilityEvidence = Evidence;
    using LimiterEvidence = Evidence;
    using AssessmentPolicy = Assessment;
    using LimiterAdapter = typename detail::AcceptedLimiterLayout<Accepted>::type;
    using Core = AMRCore<NodesPerAxis, NumberOfData, Config, Precision, Accepted, Transfer, Evidence, Assessment>;
    using Coordinates = typename Core::Coordinates;
    using WaveSpeed = typename Core::WaveSpeed;
    using PhysicalAdmissibility = typename Core::PhysicalAdmissibility;
    using InitialDataCheck = typename Core::InitialDataCheck;
    using TransferAdmissibility = typename Core::TransferAdmissibility;
    using NativeSubcells = typename Core::NativeSubcells;
    using Statistics = typename Core::Statistics;

    /** Starts the votes on the initial mesh and forgets what earlier steps stored. */
    void beginInitialisation(PhysicalAdmissibility admissible = nullptr, InitialDataCheck check = nullptr);

    /** Prepares a time step of a solver without a limiter. The criterion keeps the pointers to the solution
     * and reads the solution after the step, in command(). */
    template<typename SolutionView>
      requires (not Accepted::HasHybridSubcells)
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed = nullptr,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr);

    /** Prepares a time step of ADER-DG with the subcell limiter: the DG solution, the limiter statuses and
     * the subcell values. */
    template<typename SolutionView, typename StatusView>
      requires (Accepted::HasHybridSubcells)
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution, StatusView statuses,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr,
                   NativeSubcells subcells = {});

    /** Let the initial votes compare each cell with its neighbours once all cells hold their initial data. */
    template<typename SolutionView>
      requires (not Accepted::HasHybridSubcells)
    void bindInitialState(const peano4::CellConnectivity& mesh, SolutionView solution, const double* nodes) {
      _core.bindInitialState(mesh, solution, NoStatusView{}, nodes);
    }
    /** Computes the decisions of all cells in parallel. command() then only reads them. */
    void assess(double time) { _core.assess(time); }
    /** Switches the coarsening checks of this step on or off. Refinement is checked in every prepared step. */
    void setCoarseningAssessment(bool enabled) { _core.setCoarseningAssessment(enabled); }
    /** Returns the vote of the cell at x with width h: Refine, Keep or Erase. Q is its solution after the step. */
    [[nodiscard]] exahype2::RefinementCommand command(
      const Precision* Q, const Coordinates& x, const Coordinates& h, double time) const;
    /** Ends the step. Legacy mode stores the quiet time of every cell for the next step. */
    void finishStep();
    /** Returns what the criterion counted in this step. */
    [[nodiscard]] Statistics statistics() const { return _core.statistics(); }
    /** Sets the crossing time that legacy mode uses for a cell in which nothing moves. */
    void setReferenceTime(double time);
    /** Forgets the mesh, the solver callbacks and the quiet times. */
    void reset();

  private:
    Core _core;
  };
} // namespace exahype2::amr
#include "Criterion.cpph"
