#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
import os
import signal
import subprocess
from time import time
import pathlib
import utils
from tqdm.contrib.concurrent import process_map


@dataclass
class CompilationInfo:
    model_meta: utils.ModelMeta
    compiled: bool
    compile_time_ms: str
    stdout: str
    stderr: str


def remove_blobs_curdir():
    for file in os.listdir():
        if pathlib.Path(file).suffix == ".blob":
            os.remove(file)


def one_model(compile_tool, model_path, model_meta, device, input_precision, output_precision,  timeout_seconds, config_path):
    args = [str(compile_tool), "-d", device, "-m", str(model_path),
            "-ip", input_precision, "-op", output_precision, "-c", config_path]

    env = os.environ.copy()
    env["IE_VPUX_LOG_LEVEL"] = 'LOG_ERROR'
    env["IE_VPUX_CREATE_EXECUTOR"] = '0'

    start = time()

    try:
        output = subprocess.run(args, encoding="utf-8", env=env,
                                capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        error_msg = "Timed out after {} seconds".format(timeout_seconds)
        return CompilationInfo(model_meta=model_meta, compiled=False, compile_time_ms=timeout_seconds * 1000, stdout="", stderr=error_msg)
    except subprocess.CalledProcessError as e:
        print(e.output)

    time_delta = int((time() - start) * 1000)

    is_ok = (output.returncode == 0)

    stderr_str = output.stderr.strip()
    if output.returncode == -signal.SIGSEGV:
        stderr_str = stderr_str + "[Segmentation fault]"

    return CompilationInfo(model_meta=model_meta, compiled=is_ok, compile_time_ms=time_delta,
                           stdout=output.stdout.strip(), stderr=stderr_str)


def compile_proc_func(args):
    output_root, model_path, compile_tool, device, input_precision, output_precision, timeout_seconds, use_irs, config_path = args
    model_meta = utils.metadata_from_path(model_path)
    relative_dir = model_meta.model_path_relative
    utils.relative_change_dir(output_root, relative_dir)
    compile_info = one_model(
        compile_tool, model_path, model_meta, device, input_precision, output_precision, timeout_seconds, config_path)

    if use_irs:
        remove_blobs_curdir()

    return compile_info


def all_models(compile_tool, models_list, device, input_precision, output_precision, output_dir, timeout_seconds, use_irs, config_path):
    compile_input = [(output_dir, model_path, compile_tool,
                      device, input_precision, output_precision, timeout_seconds, use_irs, config_path) for model_path in models_list]
    return process_map(compile_proc_func, compile_input, max_workers=os.cpu_count(), chunksize=1)
