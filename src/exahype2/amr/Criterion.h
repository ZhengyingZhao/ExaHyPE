// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "AMRCore.h"
#include <type_traits>

#if defined(PEANO_WITH_GPU_BACKEND)
#error "The AMR criterion currently supports CPU execution only."
#endif

namespace exahype2::amr {
  namespace detail {
    template<typename Accepted, typename = void>
    struct AcceptedLimiterLayout { using type = void; };
    template<typename Accepted>
    struct AcceptedLimiterLayout<Accepted, std::void_t<typename Accepted::LimiterAdapter>> {
      using type = typename Accepted::LimiterAdapter;
    };
  }

  /** AMR decisions for FV, DG and DG/FV hybrid solvers. */
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

    /** Reset history and enable checks on the initial solution. */
    void beginInitialisation(PhysicalAdmissibility admissible = nullptr, InitialDataCheck check = nullptr);

    /** Bind the live solution before stepping; command() reads it after the step. */
    template<typename SolutionView>
      requires (not Accepted::HasHybridSubcells)
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed = nullptr,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr);

    /** Bind the DG/FV pools and limiter roles before stepping. */
    template<typename SolutionView, typename StatusView>
      requires (Accepted::HasHybridSubcells)
    void beginStep(const peano4::CellConnectivity& mesh, SolutionView solution, StatusView statuses,
                   const exahype2::TimeStepping& clock, const double* nodes,
                   const double* restriction, const double* prolongation, WaveSpeed waveSpeed,
                   PhysicalAdmissibility admissible = nullptr, TransferAdmissibility transferAdmissible = nullptr,
                   NativeSubcells subcells = {});

    /** Return Refine, Keep or Erase for this cell's accepted state. */
    [[nodiscard]] exahype2::RefinementCommand command(
      const Precision* Q, const Coordinates& x, const Coordinates& h, double time) const;
    /** Finish assessment and update quiet-time history in legacy mode. */
    void finishStep();
    /** Read cached assessment and vote counters. */
    [[nodiscard]] Statistics statistics() const { return _core.statistics(); }
    /** Set the fallback crossing time used when the wave speed is zero. */
    void setReferenceTime(double time);
    /** Clear cached mesh data and quiet history. */
    void reset();

  private:
    Core _core;
  };
} // namespace exahype2::amr
#include "Criterion.cpph"
