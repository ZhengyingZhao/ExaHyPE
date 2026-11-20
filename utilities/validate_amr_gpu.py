#!/usr/bin/env python3
r"""Run the AMR generator tests, C++ unit tests and a four-solver example in 2D and 3D.

Use a build from this checkout with BUILD_TESTING=ON, Asserts in
SUPPORTED_BUILD_TYPES and both dimensions in SUPPORTED_DIMENSIONS.

  python utilities/validate_amr_gpu.py --build-dir build/amr-cpu --jobs 4
  CUDA_VISIBLE_DEVICES=0 python utilities/validate_amr_gpu.py \
      --build-dir build/amr-cuda --jobs 4 --require-gpu

The build directory selects the runtime. A CPU build runs both criteria on the
host. Use --require-gpu with a native CUDA or HIP build to test on a GPU.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


def read_cache(path):
    """Return the CMake cache entries by name."""
    entries = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)$", line)
        if match:
            entries[match[1]] = match[2]
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, required=True, help="Existing configured runtime build directory.")
    parser.add_argument("--output", type=Path, help="Fresh results directory (default: build/amr-validation-<UTC time>).")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="OpenMP threads per single-rank test process.")
    parser.add_argument("--storage-precisions", nargs="+", choices=("double", "float"), default=("double", "float"),
                        help="Stored state precisions to exercise in the four-solver smoke tests.")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per command.")
    parser.add_argument("--require-gpu", action="store_true", help="Require a native CUDA/HIP runtime; reject CPU fallback backends.")
    arguments = parser.parse_args()
    if min(arguments.jobs, arguments.threads, arguments.timeout) <= 0:
        parser.error("--jobs, --threads and --timeout must be positive")

    source = Path(__file__).resolve().parents[1]
    build = arguments.build_dir.resolve()
    cache_file = build / "CMakeCache.txt"
    configuration = build / "PeanoConfig.cmake"
    if not cache_file.is_file() or not configuration.is_file():
        parser.error("--build-dir must contain CMakeCache.txt and PeanoConfig.cmake; configure the runtime first")
    cache = read_cache(cache_file)
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != source:
        parser.error("the runtime build belongs to another checkout; configure a build from this source tree")
    if cache.get("BUILD_TESTING", "").upper() not in ("ON", "TRUE", "YES", "1"):
        parser.error("the runtime must be configured with -DBUILD_TESTING=ON")
    if "Asserts" not in cache.get("SUPPORTED_BUILD_TYPES", "").split(";"):
        parser.error("the runtime must include Asserts in SUPPORTED_BUILD_TYPES")
    if not {"2", "3"}.issubset(cache.get("SUPPORTED_DIMENSIONS", "").split(";")):
        parser.error('the runtime must include both dimensions: -DSUPPORTED_DIMENSIONS="2;3"')
    match = re.search(r'set\(PEANO_WITH_GPU_BACKEND\s+"([^"]+)"\)', configuration.read_text())
    if not match:
        parser.error("cannot read the resolved accelerator backend from PeanoConfig.cmake")
    backend = match[1]
    if arguments.require_gpu and backend not in ("cuda", "hip"):
        parser.error(f"--require-gpu needs native CUDA/HIP; resolved backend is {backend!r}. Other backends may use host fallback.")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (arguments.output or source / "build" / f"amr-validation-{stamp}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be absent or empty so old generated files cannot affect this run")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source / "api") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["PEANO_BUILD_DIR"] = str(build)
    environment["OMP_NUM_THREADS"] = str(arguments.threads)
    environment["PYTHONUNBUFFERED"] = "1"
    manifest = {
        "source": str(source), "build": str(build), "backend": backend,
        "require_gpu": arguments.require_gpu, "started_utc": stamp,
        "storage_precisions": list(arguments.storage_precisions),
        "status": "running", "commands": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, command, working_directory=source):
        """Run one validation command and save its output to a named log."""
        log = output / f"{name}.log"
        print(f"{name}: {shlex.join(map(str, command))}\n  log: {log}", flush=True)
        started = time.monotonic()
        entry = {"name": name, "command": list(map(str, command)), "cwd": str(working_directory), "log": str(log)}
        manifest["commands"].append(entry)
        save()
        with log.open("w") as stream:
            stream.write(f"cwd: {working_directory}\ncommand: {shlex.join(map(str, command))}\n")
            stream.flush()
            try:
                process = subprocess.Popen(command, cwd=working_directory, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    entry["returncode"] = process.wait(timeout=arguments.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    # The process group includes any compilers started by the build.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
            except (OSError, subprocess.TimeoutExpired) as error:
                entry["error"] = str(error)
                entry["returncode"] = None
                stream.write(f"\nDriver error: {error}\n")
        entry["seconds"] = round(time.monotonic() - started, 3)
        save()
        if entry["returncode"] != 0:
            print(f"FAILED: {name}\n" + "\n".join(log.read_text(errors="replace").splitlines()[-40:]), file=sys.stderr)
            raise RuntimeError(f"{name} failed; see {log}")

    print(f"Resolved runtime backend: {backend}; results: {output}", flush=True)
    if backend == "no":
        print('CPU runtime: execution="gpu" will validate the staged host algorithm only.', flush=True)
    elif not arguments.require_gpu:
        print("Configured-backend validation; host fallback is not excluded by this invocation.", flush=True)
    save()
    try:
        run("python-api", [sys.executable, "-m", "unittest", "discover", "-s", "api/exahype2/tests", "-p", "amr_gpu.py"])
        run("build-units", ["cmake", "--build", build, "--target", "UnitTestsAsserts2D", "UnitTestsAsserts3D", "--parallel", str(arguments.jobs)])
        for dimensions in (2, 3):
            working = output / f"unit-{dimensions}d"
            working.mkdir()
            run(f"units-{dimensions}d", [build / f"UnitTestsAsserts{dimensions}D", "--junit", working / "junit.xml"], working)
            for execution in ("cpu", "gpu"):
                for precision in dict.fromkeys(arguments.storage_precisions):
                    case = f"{execution}-{dimensions}d-{precision}"
                    smoke = output / f"smoke-{case}"
                    run(f"generate-{case}", [
                        sys.executable, source / "api/exahype2/tests/amr_gpu_smoke.py",
                        "--criterion-execution", execution, "--storage-precision", precision,
                        "--build-mode", "Asserts", "--dimensions", str(dimensions),
                        "--jobs", str(arguments.jobs), "--output", smoke,
                    ])
                    run(f"smoke-{case}", [smoke / "ExaHyPE", "-ns", "0"], smoke)
    except (RuntimeError, KeyboardInterrupt) as error:
        manifest["status"] = "failed" if isinstance(error, RuntimeError) else "interrupted"
        manifest["error"] = str(error)
        save()
        print(f"Validation {manifest['status']}; results: {output / 'results.json'}", file=sys.stderr)
        return 1
    manifest["status"] = "passed"
    manifest["scope"] = "native GPU runtime execution" if arguments.require_gpu else "configured runtime execution; hardware offload not certified"
    save()
    print(f"Validation passed ({manifest['scope']}); results: {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
