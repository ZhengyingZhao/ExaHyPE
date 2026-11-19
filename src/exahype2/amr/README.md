# AMR with GPU solvers

`exahype2.amr.AMR(...).install(solver)` supports FV, ADER-DG, RKDG and
ADER-DG with the native subcell limiter in accelerator builds. Install the
criterion after configuring the solver and its limiter. The Python API and
criterion tolerances are the same for CPU and GPU builds.

The numerical solver runs on the selected accelerator. AMR assessment runs on
host threads, including neighbour checks, parent transfer proofs and quiet-time
history. This implementation does not offload the criterion itself to the GPU.

The generated solver declares `RefinementCriterionOnHost = true` and keeps
`refinementCriterion` outside device code. Grid construction, initialisation,
accepted-step voting and votes on cells created by a regrid all honour this
flag. Solvers without it retain device refinement voting.

Before each selected step the criterion binds persistent host solution pools.
After all application finish hooks, the solver downloads the accepted solution
into those same pools before voting. Limited ADER-DG also refreshes limiter
statuses and native FV subcells. Pool chunks keep their addresses during this
refresh, so cached cell pointers see the completed state. DG transfer and
limiter projection proofs use host accessor arrays rather than device table
pointers. This adds device-to-host transfers on assessment steps; no GPU
performance benefit is claimed for AMR assessment.

`assessment_interval` skips both assessment and its downloads on unselected
steps. After a regrid, the existing vote pause still refreshes the accepted
solution so legacy quiet histories can advance correctly.

For a custom callback installed with `external_callback=True`, define
`refinementCriterion` as a host function. Other PDE terms used by device
kernels still need their normal device definitions.

The existing restrictions remain: float or double stored states, complete
uncoupled solver domains, IEEE arithmetic without fast-math, and the existing
single-rank requirement for native FV/DG assessment. GPU support does not add
remote accepted-neighbour exchange.

## Regression tests

Run the generator checks from the repository root:

```sh
PYTHONPATH=api python -m unittest discover -s api/exahype2/tests -p amr_gpu.py
```

`RefinementVotesTest.cpp` covers host-only callbacks, size guards, grid and
listed-cell voting, plus refreshing accepted data without changing chunk
addresses. `SubcellLimiterTest.cpp` checks host parent admissibility and maximum
principle proofs using projection accessors with no device pointer members.
Both files are included in the GPU test sources and run in 2D and 3D.

On a CUDA machine, for example, configure a fresh build (replace `sm_80` with
the architecture of the installed device):

```sh
cmake -S . -B build/amr-gpu \
  -DPEANO_WITH_GPU_BACKEND=native -DPEANO_WITH_GPU_ARCH=sm_80 \
  -DSUPPORTED_BUILD_TYPES=Asserts '-DSUPPORTED_DIMENSIONS=2;3' \
  -DBUILD_TESTING=ON
cmake --build build/amr-gpu --target UnitTestsAsserts2D UnitTestsAsserts3D --parallel
OMP_NUM_THREADS=2 build/amr-gpu/UnitTestsAsserts2D
OMP_NUM_THREADS=2 build/amr-gpu/UnitTestsAsserts3D
```

Generate and run a four-solver integration test against that runtime:

```sh
PEANO_BUILD_DIR="$PWD/build/amr-gpu" PYTHONPATH=api \
  python api/exahype2/tests/amr_gpu_smoke.py --build-mode Asserts \
  --dimensions 2 --output build/amr-gpu-smoke-2d
OMP_NUM_THREADS=2 build/amr-gpu-smoke-2d/ExaHyPE -ns 0
```

Repeat with `--dimensions 3` and a fresh output directory. The test changes an
initially constant accepted state in a numerical kernel and requires the native
criteria to observe its new detail. The limited solver also enables PAD and DMP.

CPU builds exercise host dispatch and numerical checks, but do not establish
device compilation or execution. Run these tests with the intended accelerator
backend and hardware before using a new GPU toolchain in production.
