# AMR criterion

This directory holds the adaptive mesh refinement (AMR) criterion of ExaHyPE 2. After every time
step it decides for each cell whether the cell refines, keeps its size or coarsens together with its
siblings. It works with FV, ADER-DG and RKDG solvers and with ADER-DG with the subcell limiter, in
CPU and in GPU builds.

Every cell gets a score that measures how poorly the cell and its faces resolve the solution. A cell
refines if its score exceeds the resolution tolerance. Sibling cells coarsen if their parent
reproduces them within the transfer tolerance and is itself resolved. The code calls the decision
for a cell its vote. [AMRCore.h](AMRCore.h) explains the rules, and
[api/exahype2/amr.py](../../../api/exahype2/amr.py) explains every setting.

## Usage

Create the criterion in the Python script of the project and install it on the solver. Configure the
solver first, including its limiter. Do not set a `refinement_criterion` term: the criterion
provides it.

```python
import exahype2

project = exahype2.Project(
    namespace=["examples", "amr"], project_name="Bump", dimensions=2,
    size=[1.0] * 3, offset=[0.0] * 3, periodic_bc=[True] * 3,
    min_end_time=0.5, max_end_time=0.5,
    min_grid_levels=2, max_grid_levels=4, directory=".",
)
solver = exahype2.dg.ader.Solver(
    name="Bump", order=3, unknowns=1, auxiliary_variables=0,
    time_stepping=exahype2.TimeStepping.Adaptive(0.9),
)
solver.set_pde_terms(
    initial_condition="""
const double dx = x(0) - 0.5, dy = x(1) - 0.5;
Q[0] = std::exp(-100.0 * (dx * dx + dy * dy));
""",
    max_eigenvalue="return 1.0;",
    flux="F[0] = normal == 0 ? Q[0] : 0.0;",
)
exahype2.amr.AMR(state_scales=(1.0,), transfer_tolerance=1.0e-3).install(solver)
project.add_solver(solver)
project.generate_and_build()
```

The script moves a bump along the x axis. The mesh may use the levels 2 to 4, and the criterion
chooses the level of every cell. Save the script as `bump.py` in an empty directory and run it there
with a Peano build:

```sh
PEANO_BUILD_DIR=<Peano build directory> PYTHONPATH=<repository>/api python bump.py
./ExaHyPE
```

The log of the run reports the number of cells and the levels of the mesh whenever the criterion
has changed it.

The most important settings:

| Setting | Meaning |
|---|---|
| `state_scales` | One positive scale per stored value. Differences are divided by the scales before they are compared with the tolerances. |
| `transfer_tolerance` | The largest scaled change of the solution that coarsening may cause. |
| `resolution_tolerance` | A cell with a larger score refines. The default is `transfer_tolerance`. |
| `assessment_interval` | With N, the criterion runs in the first step and then in every N-th step. |
| `coarsening_interval` | With M, only every M-th run of the criterion checks whether cells can coarsen. |
| `execution` | `"cpu"` or `"gpu"`: where the criterion runs. See the next section. |

## Where the criterion runs

The solver computes where the build puts it: on the CPU, or on a GPU in a GPU build. The setting
`execution` selects where the criterion computes its decisions. Both values work in both kinds of
build.

After a time step the criterion reads the solution of the completed step. This is the solution after
the limiter and after the code that the application runs at the end of a step.

### `execution="cpu"`

The criterion runs on the CPU threads. This is the default.

With a solver on a GPU, the solver copies its solution to host memory before the criterion reads it.
For ADER-DG with the subcell limiter it also copies the limiter statuses and the FV subcell values.
In a CPU build the criterion reads the memory of the solver and nothing is copied.

In a step that `assessment_interval` skips, every cell keeps its size and nothing is copied.

An application can keep its own refinement function. Install the criterion with
`install(solver, external_callback=True)` and define `refinementCriterion()` in C++ as a host
function. It returns `amrCommand(Q, x, h, t)` wherever it wants the decision of the criterion.

### `execution="gpu"`

The criterion reads the solution in device memory and runs its checks in GPU kernels. Only the
decision for every cell comes back to the host.

- With `sample_initial_data`, the criterion calls `initial_condition` on the device. `install()`
  marks the term for this. An application that defines `initialCondition()` in a C++ file of its own
  has to make it callable on the device.
- `external_callback` and a `coarsening_interval` above 1 need `execution="cpu"`.

Both values of `execution` apply the same checks in a time step. They differ on the initial mesh:
with `"cpu"`, a solver without a limiter scores an initial cell together with its neighbours. With
`"gpu"`, every initial cell is judged on its own.

## Requirements

- Stored values are `float` or `double`.
- The solver covers the whole domain. `install()` rejects a solver that owns only some of the cells.
- The code is compiled without fast-math. The transfer check relies on IEEE rounding.
- The criterion does not exchange solutions between MPI ranks. A solver without a limiter stops
  with an error if it runs on more than one rank.
- `assessment_interval` and `coarsening_interval` above 1 need the default indicator
  (`resolution_indicator="prediction"`).
- `sample_initial_data` is for DG solvers whose `initial_condition` depends on the position only.

## Tests

[utilities/validate_amr_gpu.py](../../../utilities/validate_amr_gpu.py) runs the tests of the table
against one Peano build, in 2D and 3D. The build needs `-DBUILD_TESTING=ON`, `Asserts` in
`SUPPORTED_BUILD_TYPES` and both dimensions in `SUPPORTED_DIMENSIONS`. Run the script from the
repository root:

```sh
python utilities/validate_amr_gpu.py --build-dir <Peano build directory> --jobs 4
```

| Test | What it checks |
|---|---|
| `api/exahype2/tests/amr_gpu.py` | The C++ code that `install()` generates for both values of `execution`. |
| `AMRDeviceCriterionTest.cpp` | The CPU and the GPU criterion cast the same votes on small meshes. |
| `AMRDeviceSensorsTest.cpp` | The scores and the bounds of the transfer check. |
| `RefinementVotesTest.cpp` | The solver collects the same votes from a criterion on the CPU and from GPU kernels. |
| `SubcellLimiterTest.cpp` | The limiter checks of a proposed parent cell. |
| `api/exahype2/tests/amr_gpu_smoke.py` | An example with an FV, an ADER-DG, an RKDG and a limited ADER-DG solver builds and runs, for both values of `execution` and for `float` and `double`. |

The C++ tests are part of `UnitTestsAsserts2D` and `UnitTestsAsserts3D`. One more Python test,
`api/exahype2/tests/amr_host.py`, checks the settings and the code generated for `execution="cpu"`.
Run both Python tests from the repository root:

```sh
PYTHONPATH=api python -m unittest discover -s api/exahype2/tests -p 'amr*.py'
```

In a CPU build, `execution="gpu"` runs its kernels on the CPU. To test on a GPU, configure a build
with a GPU backend as [documentation/exahype2/gpus.dox](../../../documentation/exahype2/gpus.dox)
describes, and add `--require-gpu`. For an NVIDIA GPU with the architecture `sm_80`:

```sh
cmake -S . -B build/amr-gpu \
  -DPEANO_WITH_GPU_BACKEND=native -DPEANO_WITH_GPU_ARCH=sm_80 \
  -DSUPPORTED_BUILD_TYPES=Asserts '-DSUPPORTED_DIMENSIONS=2;3' -DBUILD_TESTING=ON
python utilities/validate_amr_gpu.py --build-dir build/amr-gpu --jobs 4 --require-gpu
```

## Implementation notes

- `install()` adds two flags to the generated solver. `DeferTimeStepRefinement` moves the votes
  behind the code that the application runs at the end of a step. `RefinementCriterionOnHost` is
  true for `execution="cpu"`. The solver then calls the criterion on the host during grid
  construction, initialisation, time stepping and regridding, and no GPU kernel calls it.
- With `execution="cpu"`, the criterion keeps pointers to the solution of every cell from the start
  of a step until the votes. The host copy of the solution keeps its addresses when the solver
  refreshes it, so these pointers stay valid.
- With `execution="cpu"`, the criterion reads the DG matrices and the projection tables of the
  limiter from host arrays, also in a GPU build.
- In the first step after the mesh has changed, the solver casts no votes. With `execution="cpu"`
  it still refreshes the host copy of the solution if the criterion runs in that step. The
  indicator `resolution_indicator="legacy"` needs this, because it counts in every step how long a
  cell has been resolved.
