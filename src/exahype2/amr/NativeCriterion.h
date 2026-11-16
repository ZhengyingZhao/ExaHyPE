// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once
#include "Criterion.h"

namespace exahype2::amr {
  /** Native FV/DG criterion. */
  template<int NodesPerAxis, int NumberOfData, typename Config, typename Precision = double>
  using NativeCriterion = Criterion<NodesPerAxis, NumberOfData, Config, Precision>;
} // namespace exahype2::amr
