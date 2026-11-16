// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "Criterion.h"
#include "LimiterEvidence.h"

namespace exahype2::amr {
  /** Hybrid DG/FV criterion with limiter evidence. */
  template<int NodesPerAxis, int NumberOfData, typename Config, typename Precision = double>
  using LimiterCriterion = Criterion<NodesPerAxis, NumberOfData, Config, Precision,
    HybridAcceptedSolutionAdapter<NodesPerAxis, NumberOfData, Precision,
      typename detail::LimiterAdapterFor<Config, NodesPerAxis>::type>,
    DGTransferPolicy<NodesPerAxis, NumberOfData, Config, Precision>,
    SubcellLimiterEvidence<typename detail::LimiterAdapterFor<Config, NodesPerAxis>::type>,
    LimiterCompatibilityPolicy>;
} // namespace exahype2::amr
