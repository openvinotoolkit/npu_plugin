#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
import os
import signal
import subprocess
from time import time
import utils
from tqdm.contrib.concurrent import process_map
import pathlib


@dataclass
class BenchmarkInfo:
    model_meta: utils.ModelMeta
    executed: bool
    execution_time_ms: str
    stdout: str
    stderr: str


def one_model(benchmark_tool, model_path, model_meta, device, timeout_seconds, config_path):
    # TODO config_path ignored for now since require .json instead of .txt
    # TODO add inference time measurements even for one inference. Right now it's FEIL
    nireq = '1'
    niter = '1'
    args = [str(benchmark_tool), "-d", device, "-m",
            str(model_path), "-niter", niter, "-nireq", nireq]

    env = os.environ.copy()
    env["IE_VPUX_LOG_LEVEL"] = 'LOG_ERROR'

    start = time()

    try:
        output = subprocess.run(
            args, encoding="utf-8", env=env, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        error_msg = "Timed out after {} seconds".format(timeout_seconds)
        return BenchmarkInfo(model_meta=model_meta, executed=False, execution_time_ms=timeout_seconds * 1000, stdout="", stderr=error_msg)
    except subprocess.CalledProcessError as e:
        print(e.output)

    time_delta = int((time() - start) * 1000)

    is_ok = (output.returncode == 0)

    stderr_str = output.stderr.strip()

    if output.returncode == -signal.SIGSEGV:
        stderr_str = stderr_str + "[Segmentation fault]"

    return BenchmarkInfo(model_meta=model_meta, executed=is_ok, execution_time_ms=time_delta,
                         stdout=output.stdout.strip(), stderr=stderr_str)


def benchmark_proc_func(args):
    output_root, model_path, benchmark_tool, device, timeout_seconds, config_path = args
    model_meta = utils.metadata_from_path(model_path)
    relative_dir = model_meta.model_path_relative
    utils.relative_change_dir(output_root, relative_dir)
    benchmark_info = one_model(
        benchmark_tool, model_path, model_meta, device, timeout_seconds, config_path)

    return benchmark_info


def all_models(benchmark_tool, models_list, device, output_dir, timeout_seconds, config_path):
    benchmark_input = [(output_dir, model_path, benchmark_tool,
                        device, timeout_seconds, config_path) for model_path in models_list]
    return process_map(benchmark_proc_func, benchmark_input, max_workers=1, chunksize=1)
