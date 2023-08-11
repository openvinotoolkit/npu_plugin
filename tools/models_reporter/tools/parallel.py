#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
from functools import partial
from time import time
from tqdm.contrib.concurrent import process_map
from abc import ABC, abstractmethod
import subprocess
import utils
import tools.return_code as return_code


@dataclass
class ExecutionResult(utils.StrictTypeCheck):
    return_code: int
    execution_time_ms: int
    stdout: str
    stderr: str


class ProcessFunctions(ABC):
    @abstractmethod
    def get_environment(binaries):
        pass

    @abstractmethod
    def get_argument_list(parser, plugin_config_path, model_path):
        pass

    @abstractmethod
    def cleanup(parser):
        pass


def one_model(args, env, timeout):
    args = list(map(str, args))
    start = time()

    try:
        result = subprocess.run(args, encoding="utf-8", env=env,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        error_msg = f"Timed out after {timeout} seconds"
        return ExecutionResult(return_code=1, execution_time_ms=timeout * 1000, stdout="", stderr=error_msg)
    except subprocess.CalledProcessError as e:
        return ExecutionResult(return_code=1, execution_time_ms=0, stdout="", stderr=e.output)

    time_delta = int((time() - start) * 1000)

    return ExecutionResult(return_code=result.returncode, execution_time_ms=time_delta,
                           stdout=result.stdout.strip(), stderr=result.stderr.strip())


def change_work_dir(output_dir, model_path):
    model_meta = utils.metadata_from_path(model_path)
    relative_dir = model_meta.model_path_relative
    utils.relative_change_dir(output_dir, relative_dir)


def process_func(parser, plugin_config_path, impl, timeout, model_path):
    change_work_dir(parser.output_dir, model_path)
    env = impl.get_environment(parser.binaries)
    args = impl.get_argument_list(parser, plugin_config_path, model_path)
    result = one_model(args, env, timeout)
    impl.cleanup(parser)

    if result.return_code not in [0, 1]:
        result.stderr += "\n" + return_code.decode(result.return_code)
        result.stderr = result.stderr.strip()

    return result


def all_models(parser, models_list, plugin_config_path, impl, timeout, max_workers):
    func = partial(process_func, parser, plugin_config_path, impl, timeout)
    return process_map(func, models_list, max_workers=max_workers, chunksize=1)
