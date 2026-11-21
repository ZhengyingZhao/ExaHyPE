# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""Adaptive mesh refinement for FV, ADER-DG and RKDG solvers."""

import math
import warnings
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["AMR", "Native", "Limiter", "LinearObservable"]


def _positive(name, value):
    """Validate a positive finite value and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Criterion {} must be finite and positive.".format(name))
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("Criterion {} must be finite and positive.".format(name)) from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Criterion {} must be finite and positive.".format(name))
    return value


def _finite_real(name, value):
    """Validate a finite real number and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Criterion {} must be a finite real number.".format(name))
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("Criterion {} must be a finite real number.".format(name)) from None
    if not math.isfinite(value):
        raise ValueError("Criterion {} must be a finite real number.".format(name))
    return value


@dataclass(frozen=True, kw_only=True)
class LinearObservable:
    """Set an additional AMR budget for z = sum(weights[i] * Q[i]).

    The budget is absolute_tolerance + relative_tolerance * (max(z)-min(z))
    over the accepted local data. absolute_tolerance is positive and has the
    units of z; relative_tolerance is nonnegative. Supply one weight per stored
    component, using zero for components that do not contribute to z.
    """

    weights: tuple
    absolute_tolerance: float
    relative_tolerance: float = 0.0

    def __post_init__(self):
        if not isinstance(self.weights, (list, tuple)):
            raise TypeError("LinearObservable weights must be a list or tuple.")
        weights = tuple(_finite_real("observable weight", value) for value in self.weights)
        if not weights or not any(value != 0.0 for value in weights):
            raise ValueError("LinearObservable needs finite, nonzero weights for at least one stored component.")
        absolute = _positive("observable absolute_tolerance", self.absolute_tolerance)
        relative = _finite_real("observable relative_tolerance", self.relative_tolerance)
        if relative < 0.0:
            raise ValueError("Criterion observable relative_tolerance must be nonnegative.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "relative_tolerance", relative)


def _observable_declarations(observables, values, method):
    """Give the C++ criterion the weights and error tolerances for each observable."""
    count = len(observables)
    lines = ["    static constexpr int NumberOfObservables = {};".format(count)]
    if count == 0:
        return lines
    weights = ", ".join(repr(x) for observation in observables for x in observation.weights)
    lines += [
        "    " + method + " double observableWeight(int observable, int component) {",
        "      constexpr double weights[{}]{{{}}};".format(count * values, weights),
        "      return observable >= 0 and observable < {} and component >= 0 and component < {}".format(count, values),
        "        ? weights[observable * {} + component] : std::numeric_limits<double>::quiet_NaN();".format(values),
        "    }",
    ]
    for tolerance, field in (("Absolute", "absolute_tolerance"), ("Relative", "relative_tolerance")):
        lines += [
            "    {} double observable{}Tolerance(int observable) {{".format(method, tolerance),
            "      constexpr double tolerances[{}]{{{}}};".format(
                count, ", ".join(repr(getattr(observation, field)) for observation in observables)),
            "      return observable >= 0 and observable < {} ? tolerances[observable] : std::numeric_limits<double>::quiet_NaN();".format(count),
            "    }",
        ]
    return lines


def _resolution_component_declarations(components, method):
    """Tell the prediction check which fields to measure. The default uses every field."""
    if components is None:
        return []
    return ["    " + method + " bool resolutionComponent(int component) {",
            "      return " + " or ".join("component == {}".format(c) for c in components) + ";",
            "    }"]


def _solver_adapter(solver):
    """Select the adapter for the solver's subcell limiter."""
    family = getattr(solver, "SOLVER_FAMILY", None)
    if family == "ader":
        return _ADERDGAdapter
    raise ValueError(
        "No limiter criterion adapter for solver family {!r}. "
        "The available adapter requires ADER-DG with its native subcell limiter; "
        "RKDG and standalone FV do not expose that accepted-step evidence.".format(family)
    )


def _double_quadrature_points(solver):
    """Return the generated Quadrature<double> nodes used by AMR."""
    return tuple(float(point) for point in solver.basis.rounded_quadrature_points)


def _highest_mode_weights(solver):
    """Compute reflection-symmetric weights for the highest Legendre coefficient."""
    points = tuple(Fraction.from_float(point) for point in _double_quadrature_points(solver))
    order = solver.order
    if order == 0:
        return (0.0,)
    leading = math.comb(2 * order, order)
    weights = []
    for node, point in enumerate(points):
        denominator = leading
        for other, coordinate in enumerate(points):
            if node != other:
                denominator *= point - coordinate
        weights.append(float(1 / denominator))
    parity = -1.0 if order % 2 else 1.0
    for node in range(len(points) // 2):
        weights[-1 - node] = parity * weights[node]
    if not all(math.isfinite(value) for value in weights):
        raise ValueError("The nodal basis has invalid highest-mode weights.")
    return tuple(weights)


def _taylor_weights(solver):
    """Map DG nodes to every normal Taylor coefficient on either face."""
    stored = _double_quadrature_points(solver)
    points = tuple(Fraction.from_float(1.0 - stored[-1 - node] if node < len(stored) // 2 else point)
                   for node, point in enumerate(stored))
    count = len(points)
    if (not count or len(set(points)) != count or any(point < 0 or point > 1 for point in points)
            or any(left >= right for left, right in zip(points, points[1:]))):
        raise ValueError("The nodal basis has invalid Taylor coordinates.")
    left = [0.0] * (count * count)
    for node, point in enumerate(points):
        polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for other, coordinate in enumerate(points):
            if other == node:
                continue
            product = [Fraction(0)] * (len(polynomial) + 1)
            for order, coefficient in enumerate(polynomial):
                product[order] -= coordinate * coefficient
                product[order + 1] += coefficient
            polynomial = product
            denominator *= point - coordinate
        for order, coefficient in enumerate(polynomial):
            try:
                left[order * count + node] = float(coefficient / denominator)
            except OverflowError:
                raise ValueError("The nodal basis has unbounded Taylor weights.") from None
    if not all(math.isfinite(value) for value in left):
        raise ValueError("The nodal basis has unbounded Taylor weights.")
    right = [(-1.0 if order % 2 else 1.0) * left[order * count + count - 1 - node]
             for order in range(count) for node in range(count)]
    return tuple(left + right)


def _taylor_declarations(count, weights, method):
    """Give the criterion the weights for evaluating DG Taylor coefficients at a face."""
    return [
        "    " + method + " double taylorWeight(int side, int order, int node) {",
        "      constexpr double weights[{}]{{{}}};".format(len(weights), ", ".join(repr(x) for x in weights)),
        "      return side >= 0 and side < 2 and order >= 0 and order < {0} and node >= 0 and node < {0}".format(count),
        "        ? weights[(side * {0} + order) * {0} + node] : std::numeric_limits<double>::quiet_NaN();".format(count),
        "    }",
    ]


def _transfer_weights(solver):
    """Compute outward bounds for the nodal-to-Bernstein transform.

    Mirrored entries share bounds to preserve reflection symmetry.
    """
    points = tuple(Fraction.from_float(point) for point in _double_quadrature_points(solver))
    count = len(points)
    if not count or len(set(points)) != count or any(point < 0 or point > 1 for point in points):
        raise ValueError("The nodal basis has invalid transfer coordinates.")

    def enclose(value):
        """Return floating-point bounds around an exact rational value."""
        rounded = float(value)
        if not math.isfinite(rounded):
            raise ValueError("The nodal basis has unbounded transfer weights.")
        stored = Fraction.from_float(rounded)
        return (
            math.nextafter(rounded, -math.inf) if stored > value else rounded,
            math.nextafter(rounded, math.inf) if stored < value else rounded,
        )

    lower, upper = [], []
    for node, point in enumerate(points):
        coefficients = [Fraction(1)]
        denominator = Fraction(1)
        for other, coordinate in enumerate(points):
            if other == node:
                continue
            degree = len(coefficients)
            product = []
            for entry in range(degree + 1):
                left = -coordinate * coefficients[entry] * Fraction(degree - entry, degree) if entry < degree else 0
                right = (1 - coordinate) * coefficients[entry - 1] * Fraction(entry, degree) if entry else 0
                product.append(left + right)
            coefficients = product
            denominator *= point - coordinate
        for coefficient in coefficients:
            lo, hi = enclose(coefficient / denominator)
            lower.append(lo)
            upper.append(hi)
    for entry in range(count * count // 2):
        mirror = count * count - 1 - entry
        lower[entry] = lower[mirror] = min(lower[entry], lower[mirror])
        upper[entry] = upper[mirror] = max(upper[entry], upper[mirror])
    amplification = max(
        sum((max(abs(Fraction.from_float(lower[node * count + mode])),
                 abs(Fraction.from_float(upper[node * count + mode])))
             for node in range(count)), Fraction(0))
        for mode in range(count)
    )
    return tuple(lower), tuple(upper), enclose(amplification)[1]


@dataclass(frozen=True, kw_only=True)
class _AMRSettings:
    """Choose how a solver decides which cells to refine, keep or coarsen.

    Set execution to "cpu" for host assessment or "gpu" to use the configured
    accelerator backend. Give each stored field, including auxiliaries, a
    positive scale in state_scales. transfer_tolerance limits the error allowed
    when coarsening. resolution_tolerance limits spatial detail after scaling
    and defaults to transfer_tolerance.

    With resolution_indicator="prediction", the criterion checks how well
    nearby solution profiles agree. resolution_components selects the fields
    for this check, and None uses all fields. The checks for transfer error,
    finite values and physical admissibility use the full state. linear_observables
    sets tolerances for weighted sums of fields. With assessment_interval=N, the
    solver assesses the first accepted step and then every N steps that advance
    time. On skipped steps it votes Keep. The initial mesh is assessed separately.
    With coarsening_interval=M, only every M-th assessed step checks whether
    sibling cells can coarsen. The other assessed steps still request refinement.
    Coarsening checks are the most expensive part of the CPU assessment.

    With execution="cpu", FV and DG solvers without a limiter assess the
    initial mesh with the same check as the time steps, once every cell holds
    its initial data. Other configurations check each initial cell on its own.

    With resolution_indicator="legacy", the criterion measures variation in
    each cell and across its faces. quiet_time and reference_time control how
    long a cell must stay quiet before it can coarsen.

    For DG solvers, sample_initial_data compares the initial polynomial with
    pointwise initial values on its children.
    """

    state_scales: tuple
    transfer_tolerance: float = 1.0e-3
    resolution_tolerance: float | None = None
    resolution_indicator: str = "prediction"
    quiet_time: float | None = None
    reference_time: float = 1.0
    sample_initial_data: bool = False
    linear_observables: tuple = ()
    assessment_interval: int = 1
    coarsening_interval: int = 1
    resolution_components: tuple[int, ...] | None = None
    execution: str = "cpu"

    def __post_init__(self):
        if self.execution not in ("cpu", "gpu"):
            raise ValueError("execution must be 'cpu' or 'gpu'.")
        if self.resolution_indicator not in ("prediction", "legacy"):
            raise ValueError("resolution_indicator must be 'prediction' or 'legacy'.")
        if not isinstance(self.sample_initial_data, bool):
            raise TypeError("sample_initial_data must be a boolean.")
        if isinstance(self.assessment_interval, bool) or not isinstance(self.assessment_interval, int):
            raise TypeError("assessment_interval must be an integer.")
        if not 1 <= self.assessment_interval <= 2**31 - 1:
            raise ValueError("assessment_interval must lie in [1, 2**31-1].")
        if self.assessment_interval != 1 and self.resolution_indicator != "prediction":
            raise ValueError("assessment_interval > 1 requires prediction mode; legacy quiet history needs every step.")
        if isinstance(self.coarsening_interval, bool) or not isinstance(self.coarsening_interval, int):
            raise TypeError("coarsening_interval must be an integer.")
        if not 1 <= self.coarsening_interval <= 2**31 - 1:
            raise ValueError("coarsening_interval must lie in [1, 2**31-1].")
        if self.coarsening_interval != 1 and self.resolution_indicator != "prediction":
            raise ValueError("coarsening_interval > 1 requires prediction mode.")
        if self.coarsening_interval != 1 and self.execution != "cpu":
            raise ValueError("coarsening_interval > 1 requires execution='cpu'.")
        if not isinstance(self.state_scales, (list, tuple)):
            raise TypeError("{} criterion state_scales must be a list or tuple.".format(type(self).__name__))
        object.__setattr__(
            self,
            "state_scales",
            tuple(_positive("state_scales", value) for value in self.state_scales),
        )
        if self.resolution_components is not None:
            if self.resolution_indicator != "prediction":
                raise ValueError("resolution_components requires prediction mode.")
            if not isinstance(self.resolution_components, (list, tuple)):
                raise TypeError("resolution_components must be a nonempty list/tuple of distinct stored-component indices, or None.")
            components = tuple(self.resolution_components)
            if (not components or any(isinstance(c, bool) or not isinstance(c, int)
                                      or not 0 <= c < len(self.state_scales) for c in components)
                    or len(set(components)) != len(components)):
                raise ValueError("resolution_components needs nonempty, unique indices within the stored state layout.")
            object.__setattr__(self, "resolution_components", components)
        if not isinstance(self.linear_observables, (list, tuple)):
            raise TypeError("Criterion linear_observables must be a list or tuple of LinearObservable objects.")
        observations = tuple(self.linear_observables)
        if any(not isinstance(observation, LinearObservable) for observation in observations):
            raise TypeError("Criterion linear_observables accepts only LinearObservable objects.")
        if observations and self.resolution_indicator != "prediction":
            raise ValueError("linear_observables requires resolution_indicator='prediction'.")
        if any(len(observation.weights) != len(self.state_scales) for observation in observations):
            raise ValueError("LinearObservable weights need exactly one entry per stored component, including auxiliary data.")
        object.__setattr__(self, "linear_observables", observations)
        for name in ("transfer_tolerance", "reference_time"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        quiet = self.quiet_time
        if quiet is not None and (isinstance(quiet, bool) or not isinstance(quiet, (int, float))):
            raise ValueError("Criterion quiet_time must be a finite real number.")
        if self.resolution_indicator == "legacy":
            quiet = 3.0 if quiet is None else _positive("quiet_time", quiet)
        else:
            if quiet is not None and quiet != 0.0:
                _positive("quiet_time", quiet)
                warnings.warn(
                    "quiet_time={!r} is ignored: it applies only to resolution_indicator='legacy'. "
                    "Prediction coarsening uses the current accepted state without waiting.".format(quiet),
                    UserWarning, stacklevel=3,
                )
            if self.reference_time != 1.0:
                warnings.warn(
                    "reference_time={!r} is ignored: it applies only to resolution_indicator='legacy'.".format(
                        self.reference_time),
                    UserWarning, stacklevel=3,
                )
            quiet = 0.0
        object.__setattr__(self, "quiet_time", quiet)
        object.__setattr__(self, "resolution_tolerance", self.transfer_tolerance if self.resolution_tolerance is None
                           else _positive("resolution_tolerance", self.resolution_tolerance))


def _check_refinement_callback(solver, external_callback):
    """Reject duplicate or conflicting policies before changing a solver."""
    if (getattr(solver, "_independent_native_criterion", None) is not None
            or getattr(solver, "_independent_limiter_criterion", None) is not None):
        raise ValueError("The solver already has an independent refinement criterion.")
    configured = solver.pde_terms.is_set("refinement_criterion")
    if external_callback:
        if configured and not solver.pde_terms.is_user_defined("refinement_criterion"):
            raise ValueError("external_callback cannot replace an existing inline refinement body.")
    elif configured:
        raise ValueError(
            "The solver already has a refinement criterion; use external_callback=True for an existing C++ definition."
        )


def _protect_refinement_callback(solver):
    """Reject a later refinement_criterion term. It would replace the installed policy's votes while the
    solver keeps binding and assessing the policy on every step."""
    set_pde_terms = solver.set_pde_terms

    def set_pde_terms_of_solver_with_amr(**pde_terms):
        if "refinement_criterion" in pde_terms:
            raise ValueError(
                "The installed AMR policy defines this solver's refinement criterion. Install the policy with "
                "external_callback=True to combine it with your own callback, which returns amrCommand(Q, x, h, t) "
                "wherever the policy decides."
            )
        return set_pde_terms(**pde_terms)

    solver.set_pde_terms = set_pde_terms_of_solver_with_amr


def _runtime_callback(member, operation, solver_template="", *, snapshot=False, device=False):
    """Return a C++ callback that calls a solver function with its runtime parameters."""
    coordinates = "const tarch::la::Vector<DIMENSIONS, double>&"
    arguments = {
        "WaveSpeed": ("const StoragePrecision* Q, {0} x, {0} h, double t, double dt, int normal".format(coordinates),
                      "Q, x, h, t, dt, normal"),
        "Admissible": ("const StoragePrecision* Q, int numberOfStates, {0} x, {0} h".format(coordinates),
                       "Q, numberOfStates, x, h"),
        "TransferAdmissible": ("const StoragePrecision* parent, const StoragePrecision* const* children",
                               "parent, children"),
        "InitialCondition": ("StoragePrecision* Q, {0} x, {0} h".format(coordinates), "Q, x, h"),
        "InitialDataPasses": ("const StoragePrecision* Q, {0} x, {0} h".format(coordinates), "Q, x, h"),
    }
    declaration, forwarded = arguments[operation]
    capture, parameters = ("criterionParameters", "*criterionParameters") if snapshot else (
        "parameters" if device else "&parameters", "parameters")
    return "[{}]{}({}) {{ return {}{}<{}>({}, {}); }}".format(
        capture, " GPU_CALLABLE_METHOD " if device else "", declaration,
        member, operation, solver_template, forwarded, parameters)


def _with_parameter_snapshot(body):
    """Give the CPU callbacks a shared copy of the runtime parameters for this assessment."""
    return ("{\nconst auto criterionParameters = std::make_shared<const RuntimeParameters>(parameters);\n"
            + body + "\n}\n")


def _host_operator_declarations():
    """Declare the DG nodes and parent-child transfer tables used by the CPU criterion."""
    return """
  inline static constexpr auto _amrQuadratureNodes = Quadrature<double>::getQuadraturePoints1D();
  inline static constexpr auto _amrRestriction = DGMatrices<double>::getWeightedRestrictionMatrix1D();
  inline static constexpr auto _amrProlongation = DGMatrices<double>::getProlongationMatrix1D();
"""


def _initial_data_declarations(solver, nodes, *, native, device, method):
    """Generate the DG check that compares the initial polynomial with initial values on finer cells."""
    if native:
        member, config = "_nativeCriterion", "NativeCriterionConfig"
        check = "nativeDeviceInitialDataPasses" if device else "nativeInitialDataPasses"
    else:
        member, config = "_limiterCriterion", "LimiterCriterionConfig"
        check = "deviceInitialDataPasses" if device else "initialDataPasses"
    tables = ""
    pointers = "_amrQuadratureNodes.data(), _amrProlongation.data()"
    monitor = "InitialDataMonitor"
    if device:
        tables = ("    constexpr auto deviceNodes = Quadrature<double>::getQuadraturePoints1D();\n"
                  "    constexpr auto deviceProlongation = DGMatrices<double>::getProlongationMatrix1D();\n")
        pointers = "deviceNodes.data(), deviceProlongation.data()"
        monitor = "DeviceInitialDataMonitor"
    return """
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s void %(member)sInitialCondition(
    %(storage)s* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const RuntimeParameters& parameters
  ) {
    static_assert(::exahype2::terms::HasInitialCondition<Solver, %(storage)s>);
    static_assert(not ::exahype2::terms::InitialisesPatches<Solver>);
    Solver::initialCondition(Q, x, h, parameters);
  }
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s bool %(member)sInitialDataPasses(
    const %(storage)s* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const RuntimeParameters& parameters
  ) {
%(tables)s    return ::exahype2::amr::%(check)s<%(nodes)d, %(values)d, %(config)s>(
      Q, x, h, %(pointers)s,
      %(initial_condition)s, %(admissible)s,
      ::exahype2::amr::observation_detail::%(monitor)s<
        %(nodes)d, %(values)d, %(config)s, %(storage)s>{}
    );
  }
""" % {"method": method, "member": member, "config": config, "nodes": nodes,
       "storage": solver.precision.storage, "values": solver.state.number_of_values,
       "tables": tables, "check": check, "pointers": pointers, "monitor": monitor,
       "initial_condition": _runtime_callback(member, "InitialCondition", "Solver", device=device),
       "admissible": _runtime_callback(member, "Admissible", "Solver", device=device)}


def _refresh_accepted_solution(member):
    """Give the CPU criterion the latest limiter status and FV subcell values."""
    refresh = ""
    if member == "_limiterCriterion":
        refresh = """
{
  auto& criterionLimiter = const_cast<typename SolverFamily::LimiterType&>(getLimiter());
  (void)criterionLimiter.downloadStatuses();
  (void)criterionLimiter.downloadSubcells();
}
"""
    return refresh


def _device_callbacks(policy, solver, adapter, member):
    """Let AMR kernels call the solver's PDE functions with a copy of its runtime parameters."""
    storage = solver.precision.storage
    transfer = ("return " + member + "TransferAdmissible<Solver>(parent, children, parameters);"
                if adapter is _ADERDGAdapter else "return true;")
    initial = ("return " + member + "InitialDataPasses<Solver>(Q, x, h, parameters);"
               if policy.sample_initial_data else "return true;")
    statuses = "getLimiter().getStatuses()" if adapter is _ADERDGAdapter else "::exahype2::amr::NoStatusView{}"
    subcells = ("getLimiter().getSubcells()" if adapter is _ADERDGAdapter else
                "typename ::tarch::accelerator::ChunkedMemoryPool<StoragePrecision>::ConstView{}")
    return """
  template <typename Solver>
  struct %(member)sDeviceCallbacks final {
    RuntimeParameters parameters;
    static constexpr bool HasInitialDataCheck = %(sampling)s;
    GPU_CALLABLE_METHOD double waveSpeed(
      const %(storage)s* Q, const tarch::la::Vector<DIMENSIONS,double>& x,
      const tarch::la::Vector<DIMENSIONS,double>& h, double t, double dt, int normal
    ) const { return %(member)sWaveSpeed<Solver>(Q, x, h, t, dt, normal, parameters); }
    GPU_CALLABLE_METHOD bool admissible(
      const %(storage)s* Q, int numberOfStates, const tarch::la::Vector<DIMENSIONS,double>& x,
      const tarch::la::Vector<DIMENSIONS,double>& h
    ) const { return %(member)sAdmissible<Solver>(Q, numberOfStates, x, h, parameters); }
    GPU_CALLABLE_METHOD bool transferAdmissible(
      const %(storage)s* parent, const %(storage)s* const* children
    ) const { %(transfer)s }
    GPU_CALLABLE_METHOD bool initialDataPasses(
      const %(storage)s* Q, const tarch::la::Vector<DIMENSIONS,double>& x,
      const tarch::la::Vector<DIMENSIONS,double>& h
    ) const { %(initial)s }
  };
  // Assess the solution on the device. If castVotes is true, submit mesh requests for the selected cells.
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  void assessAMROnDevice(
    const ::peano4::CellConnectivity& mesh,
    typename ::tarch::accelerator::ChunkedMemoryPool<StoragePrecision>::ConstView solution,
    const ::exahype2::TimeStepping& clock, ::exahype2::RefinementRequests& requests,
    const RuntimeParameters& parameters, ::tarch::Span<const int> cells, bool castVotes=true
  ) {
    %(member)s.assess(mesh, solution, %(statuses)s, %(subcells)s, clock, requests, cells,
      %(member)sDeviceCallbacks<Solver>{parameters}, castVotes);
  }
""" % {"member": member, "storage": storage, "transfer": transfer, "initial": initial,
       "sampling": "true" if policy.sample_initial_data else "false", "statuses": statuses, "subcells": subcells}


def _device_begin_step(adapter, member):
    """Provide the device criterion with this step's mesh, clock and any DG transfer tables."""
    tables = "nullptr, nullptr, nullptr" if adapter is _NativeFVAdapter else (
        "_amrQuadratureNodes.data(), _amrRestriction.data(), _amrProlongation.data(),\n"
        "  ::exahype2::amr::DeviceDGOperators{Quadrature<double>::QuadraturePoints1D,\n"
        "    DGMatrices<double>::WeightedRestrictionMatrix1D, DGMatrices<double>::ProlongationMatrix1D}")
    return """
static_assert(not ::exahype2::terms::HasCellPredicate<::{{FULL_QUALIFIED_SOLVER_NAME}}>,
  "The GPU AMR criterion requires a complete, uncoupled solver domain.");
%(member)s.beginStep(cellConnectivity, getTimeStepping(), %(tables)s);
""" % {"member": member, "tables": tables}


def _install_policy(policy, solver, adapter, declarations, *, member, header, policy_attribute,
                    external_callback, additional_headers=()):
    """Attach the criterion to the solver for mesh initialisation and time stepping."""
    device = policy.execution == "gpu"
    if device and external_callback:
        raise ValueError("execution='gpu' cannot use external_callback=True: the device backend owns the complete AMR assessment.")
    if device and policy.sample_initial_data:
        solver.pde_terms.set_device_callable("initial_condition")
    begin_step = _device_begin_step(adapter, member) if device else adapter.begin_step()
    if device:
        header = "exahype2/amr/DeviceCriterion.h"
        declarations += _device_callbacks(policy, solver, adapter, member)
    solver.add_includes(
        header,
        *additional_headers,
        "exahype2/terms/CellPredicate.h",
        "exahype2/terms/MaxEigenvalue.h",
        "exahype2/terms/PhysicalAdmissibility.h",
    )
    if policy.sample_initial_data:
        solver.add_includes("exahype2/terms/InitialCondition.h", "exahype2/terms/InitialisesPatches.h")
    solver.add_declarations(declarations)
    if not device:
        solver.add_declarations("""
  /** The vote of the installed AMR policy for one cell. An application that installs the policy with
   *  external_callback=True returns this vote from its refinementCriterion() wherever the policy decides. */
  static ::exahype2::RefinementCommand amrCommand(
    const %(storage)s* Q, const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h, double t
  ) { return %(member)s.command(Q, x, h, t); }
""" % {"storage": solver.precision.storage, "member": member})
    if policy.resolution_indicator == "legacy" and not solver.pde_terms.is_set("max_eigenvalue"):
        warnings.warn(
            "The legacy indicator counts quiet time in cell crossing times. This solver has no max_eigenvalue "
            "term, so every cell uses reference_time instead and coarsens after quiet_time * reference_time = "
            "{!r} time units at the earliest.".format(policy.quiet_time * policy.reference_time),
            UserWarning, stacklevel=4,
        )
    counter, selected = member + "AcceptedSteps", member + "AssessThisStep"
    coarsening = member + "CoarseningAssessments"
    reset = "\n" + member + ".reset();\n"
    refresh = "" if device else _refresh_accepted_solution(member)
    count_coarsening = ""
    if policy.coarsening_interval > 1:
        solver.add_declarations("inline static unsigned long long " + coarsening + "=0;\n")
        reset += coarsening + "=0;\n"
        begin_step += ("\n" + member + ".setCoarseningAssessment(" + coarsening + " % "
                       + str(policy.coarsening_interval) + " == 0);\n")
        count_coarsening = "if(getTimeStepping().getTimeStepSizeOfLastStep()>0.)++" + coarsening + ";\n"
    # The host votes share cell and family evidence. Evaluating it in parallel stages first keeps the votes
    # from waiting on each other. A device solution only reaches the host inside assessRefinement(), so those
    # builds evaluate the evidence within the votes.
    stages = "" if device else (
        "if (this->getSolution().isHostAccessible()) " + member + ".assess(getTimeStepping().getTimeStamp());\n")
    assessment = (refresh + "\n" + stages + "this->assessRefinement(cellConnectivity, requests, parameters);\n"
                  + member + ".finishStep();\n" + count_coarsening)
    if policy.assessment_interval > 1:
        solver.add_declarations("inline static unsigned long long " + counter + "=0;\n"
                                "inline static bool " + selected + "=true;\n")
        reset += counter + "=0;" + selected + "=true;\n"
        begin_step = (selected + "=(" + counter + " % " + str(policy.assessment_interval) + " == 0);\n"
                      "if(" + selected + "){\n" + begin_step + "\n}\n")
        assessment = ("if(" + selected + "){\n" + refresh + stages + "}\n"
                      "this->assessRefinement(cellConnectivity, requests, parameters, " + selected + ");\n"
                      "if(" + selected + "){\n" + member + ".finishStep();\n" + count_coarsening + "}\n"
                      "if(getTimeStepping().getTimeStepSizeOfLastStep()>0.)++" + counter + ";\n")
    solver.on_start_simulation(reset)
    samples = ", " + _runtime_callback(member, "InitialDataPasses", snapshot=True) if policy.sample_initial_data else ""
    # The native criteria also receive the initial mesh and the storage of its cells. The votes on the initial
    # data then compare each cell with its neighbours, as the votes of a time step do.
    initial_state = ""
    if adapter is _NativeFVAdapter:
        initial_state = member + ".bindInitialState(cellConnectivity, this->downloadSolution().getView(), nullptr);\n"
    elif adapter is _NativeDGAdapter:
        initial_state = (member + ".bindInitialState(cellConnectivity, this->downloadSolution().getView(), "
                         "_amrQuadratureNodes.data());\n")
    solver._refinement_initialisation_implementation = (
        "\n" + member + ".beginInitialisation();\n" if device else _with_parameter_snapshot(
            "\n" + member + ".beginInitialisation(" + _runtime_callback(member, "Admissible", snapshot=True)
            + samples + ");\n" + initial_state
        )
    )
    solver.on_start_time_step(begin_step)
    # User hooks can still change the solution that AMR will assess.
    solver._refinement_assessment_implementation = assessment
    solver.on_finish_simulation("\n" + member + ".reset();\n")
    solver.set_pde_terms(refinement_criterion=True if external_callback else (
        "return ::exahype2::RefinementCommand::Keep;" if device else
        "return {{SOLVER_NAME}}Base::" + member + ".command(Q, x, h, t);"
    ))
    if not device:
        solver.pde_terms.set_host_only("refinement_criterion")
    if not external_callback:
        _protect_refinement_callback(solver)
    setattr(solver, policy_attribute, policy)
    return solver


@dataclass(frozen=True, kw_only=True)
class Limiter(_AMRSettings):
    """AMR for ADER-DG with its subcell limiter.

    Read the active DG or FV solution. Limiter roles add refinement requests
    and coarsening guards to the shared AMR checks.
    """

    def install(self, solver, *, external_callback=False):
        """Attach the criterion. For CPU assessment, external_callback=True keeps the application's own
        refinementCriterion(). That callback returns amrCommand(Q, x, h, t) wherever this policy decides,
        for example after its own rules for the initial mesh. Without external_callback, the policy owns
        the refinement_criterion term and later changes of it are rejected."""
        if not isinstance(external_callback, bool):
            raise TypeError("external_callback must be a boolean.")
        adapter = _solver_adapter(solver)
        adapter.validate(solver)
        if getattr(solver, "owns_cells", False):
            raise ValueError("Limiter criterion does not support restricted or coupled solver domains.")
        if solver.precision.storage not in ("float", "double"):
            raise ValueError("Limiter criterion supports float or double stored states.")
        values = solver.state.number_of_values
        if len(self.state_scales) != values:
            raise ValueError(
                "Limiter criterion state_scales needs exactly {} stored-component scales, including auxiliary data.".format(
                    values
                )
            )
        if self.sample_initial_data:
            if getattr(solver, "_initialise_patches", False):
                raise ValueError("Initial-data sampling requires a deterministic pointwise initial condition.")
            if not solver.pde_terms.is_set("initial_condition"):
                raise ValueError("Initial-data sampling requires the solver's initial condition.")
        _check_refinement_callback(solver, external_callback)
        return _install_policy(
            self, solver, adapter, self._declarations(solver),
            member="_limiterCriterion", header="exahype2/amr/Criterion.h",
            policy_attribute="_independent_limiter_criterion", external_callback=external_callback,
            additional_headers=("exahype2/amr/LimiterEvidence.h",),
        )

    @staticmethod
    def _highest_mode_weights(solver):
        return _highest_mode_weights(solver)

    @staticmethod
    def _transfer_weights(solver):
        return _transfer_weights(solver)

    def _declarations(self, solver):
        """Give the C++ criterion the settings and physical checks for this limited solver."""
        device = self.execution == "gpu"
        method = "GPU_CALLABLE_METHOD static" if device else "static"
        scales = ", ".join(repr(value) for value in self.state_scales)
        modal_weights = ", ".join(repr(value) for value in self._highest_mode_weights(solver))
        lower, upper, amplification = self._transfer_weights(solver)
        declarations = """
  static constexpr bool DeferTimeStepRefinement = true;
  static constexpr bool RefinementCriterionOnHost = %(on_host)s;
  struct LimiterCriterionConfig final {
    // PatchSize is the limiter's number of subcells per axis.
    using LimiterAdapter = ::exahype2::amr::NativeSubcellLimiter<PatchSize>;
    static constexpr double TransferTolerance = %(transfer)r;
    static constexpr double ResolutionTolerance = %(resolution)r;
    static constexpr bool UsePredictionCriterion = %(prediction)s;
    static constexpr double QuietTime = %(quiet)r;
    static constexpr double ReferenceTime = %(reference)r;
    static constexpr bool ReuseQuietProgress = %(reuse_quiet)s;
    %(method)s double stateScale(int component) {
      constexpr double scales[%(values)d]{%(scales)s};
      return component >= 0 and component < %(values)d
        ? scales[component] : std::numeric_limits<double>::quiet_NaN();
    }
%(observables)s
%(resolution_components)s
    %(method)s double highestModeWeight(int node) {
      // Each weight gives one node's contribution to the highest Legendre coefficient.
      constexpr double weights[%(nodes)d]{%(modal_weights)s};
      return node >= 0 and node < %(nodes)d
        ? weights[node] : std::numeric_limits<double>::quiet_NaN();
    }
%(taylor)s
    static constexpr double TransferAmplification = %(amplification)r;
    // Each pair of bounds encloses one weight of the nodal-to-Bernstein transform.
    %(method)s double transferWeightLower(int node, int mode) {
      constexpr double weights[%(transfer_weights)d]{%(lower)s};
      return weights[node * %(nodes)d + mode];
    }
    %(method)s double transferWeightUpper(int node, int mode) {
      constexpr double weights[%(transfer_weights)d]{%(upper)s};
      return weights[node * %(nodes)d + mode];
    }
  };
  inline static ::exahype2::amr::%(criterion)s<
    %(nodes)d, %(values)d, LimiterCriterionConfig, %(storage)s,
    ::exahype2::amr::HybridAcceptedSolutionAdapter<
      %(nodes)d, %(values)d, %(storage)s, LimiterCriterionConfig::LimiterAdapter
    >,
    ::exahype2::amr::DGTransferPolicy<%(nodes)d, %(values)d, LimiterCriterionConfig, %(storage)s>,
    ::exahype2::amr::SubcellLimiterEvidence<LimiterCriterionConfig::LimiterAdapter>,
    ::exahype2::amr::LimiterCompatibilityPolicy
  > _limiterCriterion{};
  static auto getAMRStatistics() { return _limiterCriterion.statistics(); }

  // The quiet-time check uses the wave speed normal to a cell face.
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s double _limiterCriterionWaveSpeed(
    const %(storage)s* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    double t, double dt, int normal, const RuntimeParameters& parameters
  ) {
    if constexpr (::exahype2::terms::HasMaxEigenvalue<Solver, %(compute)s>) {
      %(compute)s state[%(values)d];
      for (int component = 0; component < %(values)d; component++) {
        state[component] = static_cast<%(compute)s>(Q[component]);
      }
      return Solver::maxEigenvalue(state, x, h, t, dt, normal, parameters);
    } else {
      return 0.0;
    }
  }
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s bool _limiterCriterionAdmissible(
    const %(storage)s* Q, int numberOfStates,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const RuntimeParameters& parameters
  ) {
    if (numberOfStates == ::exahype2::dg::getNodesPerCell(%(nodes)d)) {
      return ::exahype2::amr::parentPhysicallyAdmissible<Solver>(Q, numberOfStates, x, h, parameters);
    }
    // Active FV states use the application's physical admissibility check.
    if constexpr (::exahype2::terms::HasPhysicalAdmissibility<Solver, %(storage)s>) {
      return Solver::isPhysicallyAdmissible(Q, numberOfStates, x, h, parameters);
    } else {
      return true;
    }
  }
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s bool _limiterCriterionTransferAdmissible(
    const %(storage)s* parent, const %(storage)s* const* children,
    const RuntimeParameters& parameters
  ) {
    // The proposed parent's observables must stay within the children's bounds, up to the limiter tolerance.
    return ::exahype2::amr::parentRespectsMaximumPrinciple<Solver>(
      parent, children, %(dmp_relaxation)r, %(dmp_scaling)r, parameters
    );
  }
""" % {
            "method": method,
            "on_host": "false" if device else "true",
            "criterion": "DeviceCriterion" if device else "Criterion",
            "transfer": self.transfer_tolerance,
            "resolution": self.resolution_tolerance,
            "prediction": "true" if self.resolution_indicator == "prediction" else "false",
            "quiet": self.quiet_time,
            "reference": self.reference_time,
            "reuse_quiet": "true",
            "values": solver.state.number_of_values,
            "nodes": solver.order + 1,
            "scales": scales,
            "observables": "\n".join(_observable_declarations(self.linear_observables, solver.state.number_of_values, method)),
            "resolution_components": "\n".join(_resolution_component_declarations(self.resolution_components, method)),
            "modal_weights": modal_weights,
            "taylor": "\n".join(_taylor_declarations(solver.order + 1, _taylor_weights(solver), method)),
            "transfer_weights": len(lower),
            "lower": ", ".join(repr(value) for value in lower),
            "upper": ", ".join(repr(value) for value in upper),
            "amplification": amplification,
            "dmp_relaxation": float(solver._subcell_limiter_dmp_relaxation_parameter),
            "dmp_scaling": float(solver._subcell_limiter_dmp_differences_scaling),
            "storage": solver.precision.storage,
            "compute": solver.precision.compute,
        }
        if self.sample_initial_data:
            declarations += _initial_data_declarations(solver, solver.order + 1, native=False, device=device, method=method)
        return _host_operator_declarations() + declarations

    @staticmethod
    def _begin_step():
        """Generate the call that supplies mesh and limiter data for the next step."""
        return _ADERDGAdapter.begin_step()


class _ADERDGAdapter:
    """Supply ADER-DG states, subcells and limiter roles to AMR."""

    @staticmethod
    def validate(solver):
        """Check that the limiter and DG nodes match the adapter."""
        if not getattr(solver, "subcell_limiter_is_enabled", False):
            raise ValueError("Attach the original subcell limiter before installing its criterion.")
        order = solver.order
        points = _double_quadrature_points(solver)
        if isinstance(order, bool) or not isinstance(order, int) or order < 0 or len(points) != order + 1:
            raise ValueError("The limiter criterion requires the solver's complete DG nodal layout.")
        if any(not math.isfinite(point) or point < 0.0 or point > 1.0 for point in points) or any(
            left >= right for left, right in zip(points, points[1:])
        ):
            raise ValueError("The limiter criterion requires ordered, distinct DG nodes in [0, 1].")
        if any(abs(left + right - 1.0) > 4.0 * math.ulp(1.0) for left, right in zip(points, reversed(points))):
            raise ValueError("The limiter criterion requires a symmetric DG nodal basis.")

    @staticmethod
    def begin_step():
        """Give CPU assessment access to the DG solution, limiter status and accepted FV subcells."""
        return _with_parameter_snapshot("""
static_assert(
  not ::exahype2::terms::HasCellPredicate<::{{FULL_QUALIFIED_SOLVER_NAME}}>,
  "The independent limiter criterion requires a complete, uncoupled solver domain."
);
auto& criterionLimiter = const_cast<typename SolverFamily::LimiterType&>(getLimiter());
_limiterCriterion.beginStep(
  cellConnectivity, this->downloadSolution().getView(), criterionLimiter.downloadStatuses().getView(), getTimeStepping(),
  _amrQuadratureNodes.data(), _amrRestriction.data(), _amrProlongation.data(),
  %(wave_speed)s, %(admissible)s,
  %(transfer_admissible)s, criterionLimiter.downloadSubcells().getView()
);
""" % {"wave_speed": _runtime_callback("_limiterCriterion", "WaveSpeed", snapshot=True),
       "admissible": _runtime_callback("_limiterCriterion", "Admissible", snapshot=True),
       "transfer_admissible": _runtime_callback("_limiterCriterion", "TransferAdmissible", snapshot=True)})


def _native_solver_adapter(solver):
    """Select FV cell averages or DG nodal values for a solver without a limiter."""
    family = getattr(solver, "SOLVER_FAMILY", None)
    if family == "fv":
        return _NativeFVAdapter
    if family in ("ader", "rk"):
        return _NativeDGAdapter
    raise ValueError("No native criterion adapter for solver family {!r}.".format(family))


class _NativeDGLayout:
    """Provide the same node interface for ADER-DG and RKDG."""

    def __init__(self, solver):
        self.basis = solver.basis
        self.order = len(_double_quadrature_points(solver)) - 1


@dataclass(frozen=True, kw_only=True)
class Native(_AMRSettings):
    """AMR for FV, ADER-DG or RKDG without a subcell limiter.

    Assess the accepted solution after the completed step and application finish hooks.
    Use one MPI rank. DG initial-data sampling requires a deterministic pointwise
    initial condition.
    """

    def _validate_solver(self, solver):
        """Check the state layout and supported solver options."""
        adapter = _native_solver_adapter(solver)
        adapter.validate(solver)
        if getattr(solver, "subcell_limiter_is_enabled", False):
            raise ValueError("Native criterion requires a solver without a subcell limiter; use amr.Limiter.")
        if getattr(solver, "owns_cells", False):
            raise ValueError("Native criterion does not support restricted or coupled solver domains.")
        if solver.precision.storage not in ("float", "double"):
            raise ValueError("Native criterion supports float or double stored states.")
        values = solver.state.number_of_values
        if len(self.state_scales) != values:
            raise ValueError(
                "Native criterion state_scales needs exactly {} stored-component scales, including auxiliary data.".format(values)
            )
        if self.sample_initial_data:
            if adapter is _NativeFVAdapter:
                raise ValueError("Initial-data resampling for FV means is not supported; use the stored native means.")
            if getattr(solver, "_initialise_patches", False) or not solver.pde_terms.is_set("initial_condition"):
                raise ValueError("Initial-data sampling requires a deterministic pointwise initial condition.")
        return adapter

    def install(self, solver, *, external_callback=False):
        """Attach the criterion. For CPU assessment, external_callback=True keeps the application's own
        refinementCriterion(). That callback returns amrCommand(Q, x, h, t) wherever this policy decides,
        for example after its own rules for the initial mesh. Without external_callback, the policy owns
        the refinement_criterion term and later changes of it are rejected."""
        if not isinstance(external_callback, bool):
            raise TypeError("external_callback must be a boolean.")
        adapter = self._validate_solver(solver)
        _check_refinement_callback(solver, external_callback)
        return _install_policy(
            self, solver, adapter, self._declarations(solver, adapter),
            member="_nativeCriterion", header="exahype2/amr/Criterion.h",
            policy_attribute="_independent_native_criterion", external_callback=external_callback,
        )

    def _declarations(self, solver, adapter):
        """Give the C++ criterion this solver's field scales, tolerances and physical checks."""
        device = self.execution == "gpu"
        method = "GPU_CALLABLE_METHOD static" if device else "static"
        criterion = "DeviceCriterion" if device else "Criterion"
        count = adapter.count(solver)
        values = solver.state.number_of_values
        storage, compute = solver.precision.storage, solver.precision.compute
        lines = [
            "\n  static constexpr bool DeferTimeStepRefinement = true;",
            "  static constexpr bool RefinementCriterionOnHost = {};".format("false" if device else "true"),
            "  struct NativeCriterionConfig final {",
            "    static constexpr bool ReuseQuietProgress = true;",
            "    static constexpr bool IsFiniteVolume = {};".format("true" if adapter is _NativeFVAdapter else "false"),
            "    static constexpr double TransferTolerance = {!r};".format(self.transfer_tolerance),
            "    static constexpr double ResolutionTolerance = {!r};".format(self.resolution_tolerance),
            "    static constexpr bool UsePredictionCriterion = {};".format(
                "true" if self.resolution_indicator == "prediction" else "false"),
            "    static constexpr double QuietTime = {!r};".format(self.quiet_time),
            "    static constexpr double ReferenceTime = {!r};".format(self.reference_time),
            "    " + method + " double stateScale(int component) {",
            "      constexpr double scales[{}]{{{}}};".format(values, ", ".join(repr(x) for x in self.state_scales)),
            "      return component >= 0 and component < {} ? scales[component] : std::numeric_limits<double>::quiet_NaN();".format(values),
            "    }",
        ]
        lines += _observable_declarations(self.linear_observables, values, method)
        lines += _resolution_component_declarations(self.resolution_components, method)
        if adapter is _NativeDGAdapter:
            layout = _NativeDGLayout(solver)
            weights = _highest_mode_weights(layout)
            lower, upper, amplification = _transfer_weights(layout)
            lines += [
                "    " + method + " double highestModeWeight(int node) {",
                "      constexpr double weights[{}]{{{}}};".format(count, ", ".join(repr(x) for x in weights)),
                "      return node >= 0 and node < {} ? weights[node] : std::numeric_limits<double>::quiet_NaN();".format(count),
                "    }",
                "    static constexpr double TransferAmplification = {!r};".format(amplification),
            ]
            lines += _taylor_declarations(count, _taylor_weights(layout), method)
            for name, table in (("Lower", lower), ("Upper", upper)):
                lines += [
                    "    {} double transferWeight{}(int node, int mode) {{".format(method, name),
                    "      constexpr double weights[{}]{{{}}};".format(len(table), ", ".join(repr(x) for x in table)),
                    "      return weights[node * {} + mode];".format(count),
                    "    }",
                ]
        lines += [
            "  };",
            "  inline static ::exahype2::amr::{}<{}, {}, NativeCriterionConfig, {}> _nativeCriterion{{}};".format(
                criterion, count, values, storage),
            "  static auto getAMRStatistics() { return _nativeCriterion.statistics(); }",
        ]
        declarations = "\n".join(lines) + "\n" + """
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s double _nativeCriterionWaveSpeed(
    const %(storage)s* Q,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    double t, double dt, int normal, const RuntimeParameters& parameters
  ) {
    if constexpr (::exahype2::terms::HasMaxEigenvalue<Solver, %(compute)s>) {
      %(compute)s state[%(values)d];
      for (int component = 0; component < %(values)d; component++) {
        state[component] = static_cast<%(compute)s>(Q[component]);
      }
      return Solver::maxEigenvalue(state, x, h, t, dt, normal, parameters);
    } else {
      return 0.0;
    }
  }
  template <typename Solver = ::{{FULL_QUALIFIED_SOLVER_NAME}}>
  %(method)s bool _nativeCriterionAdmissible(
    const %(storage)s* Q, int numberOfStates,
    const tarch::la::Vector<DIMENSIONS, double>& x,
    const tarch::la::Vector<DIMENSIONS, double>& h,
    const RuntimeParameters& parameters
  ) {
    if constexpr (::exahype2::terms::HasPhysicalAdmissibility<Solver, %(storage)s>) {
      return Solver::isPhysicallyAdmissible(Q, numberOfStates, x, h, parameters);
    } else {
      return true;
    }
  }
""" % {"method": method, "storage": storage, "compute": compute, "values": values}
        if self.sample_initial_data:
            declarations += _initial_data_declarations(solver, count, native=True, device=device, method=method)
        if adapter is _NativeDGAdapter:
            declarations = _host_operator_declarations() + declarations
        return declarations


class _NativeFVAdapter:
    """Supply FV cell averages to AMR."""
    @staticmethod
    def validate(solver):
        count = getattr(solver, "_patch_size", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("The native criterion requires a complete FV patch layout.")

    @staticmethod
    def count(solver):
        return solver._patch_size

    @staticmethod
    def begin_step():
        return _with_parameter_snapshot("""
static_assert(
  not ::exahype2::terms::HasCellPredicate<::{{FULL_QUALIFIED_SOLVER_NAME}}>,
  "The native criterion requires a complete, uncoupled solver domain."
);
_nativeCriterion.beginStep(
  cellConnectivity, this->downloadSolution().getView(), getTimeStepping(),
  nullptr, nullptr, nullptr, %(wave_speed)s, %(admissible)s
);
""" % {"wave_speed": _runtime_callback("_nativeCriterion", "WaveSpeed", snapshot=True),
       "admissible": _runtime_callback("_nativeCriterion", "Admissible", snapshot=True)})


class _NativeDGAdapter:
    """Supply ADER-DG or RKDG nodal values to AMR."""
    @staticmethod
    def validate(solver):
        order = solver.order if solver.SOLVER_FAMILY == "ader" else solver._dg_order
        points = _double_quadrature_points(solver)
        if isinstance(order, bool) or not isinstance(order, int) or order < 0 or len(points) != order + 1:
            raise ValueError("The native criterion requires the solver's complete DG nodal layout.")
        if any(not math.isfinite(point) or point < 0.0 or point > 1.0 for point in points) or any(
            left >= right for left, right in zip(points, points[1:])
        ):
            raise ValueError("The native criterion requires ordered, distinct DG nodes in [0, 1].")
        if any(abs(left + right - 1.0) > 4.0 * math.ulp(1.0) for left, right in zip(points, reversed(points))):
            raise ValueError("The native criterion requires a symmetric DG nodal basis.")

    @staticmethod
    def count(solver):
        return len(_double_quadrature_points(solver))

    @staticmethod
    def begin_step():
        return _with_parameter_snapshot("""
static_assert(
  not ::exahype2::terms::HasCellPredicate<::{{FULL_QUALIFIED_SOLVER_NAME}}>,
  "The native criterion requires a complete, uncoupled solver domain."
);
_nativeCriterion.beginStep(
  cellConnectivity, this->downloadSolution().getView(), getTimeStepping(),
  _amrQuadratureNodes.data(), _amrRestriction.data(), _amrProlongation.data(),
  %(wave_speed)s, %(admissible)s
);
""" % {"wave_speed": _runtime_callback("_nativeCriterion", "WaveSpeed", snapshot=True),
       "admissible": _runtime_callback("_nativeCriterion", "Admissible", snapshot=True)})


@dataclass(frozen=True, kw_only=True)
class AMR(_AMRSettings):
    """Attach AMR using the adapter for the configured solver.

    Configure the solver and attach any subcell limiter before calling install().
    Use Native or Limiter to select the adapter explicitly.
    """

    def install(self, solver, *, external_callback=False):
        """Select the FV/DG or subcell-limiter adapter and install its callbacks."""
        policy_type = Limiter if getattr(solver, "subcell_limiter_is_enabled", False) else Native
        with warnings.catch_warnings():
            # The settings were checked, and their warnings issued, when this object was created.
            warnings.simplefilter("ignore", UserWarning)
            policy = policy_type(**{
                name: getattr(self, name) for name in _AMRSettings.__dataclass_fields__
            })
        policy.install(solver, external_callback=external_callback)
        solver._independent_amr_criterion = self
        return solver
