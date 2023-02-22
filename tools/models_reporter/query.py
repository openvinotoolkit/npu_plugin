#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pathlib
from dataclasses import dataclass
import subprocess
import utils
from tqdm.contrib.concurrent import process_map


@dataclass
class QueryInfo:
    model_meta: utils.ModelMeta
    stdout: str
    stderr: str


def one_model(query_model, model_path, device, model_meta):
    args = [str(query_model), "-m", str(model_path), "-d", device]
    timeout_seconds = 60  # just in case

    env = os.environ.copy()
    env["VPUX_SERIALIZE_CANONICAL_MODEL"] = "1"
    env["IE_VPUX_LOG_LEVEL"] = 'LOG_ERROR'
    env["IE_VPUX_CREATE_EXECUTOR"] = '0'

    try:
        output = subprocess.run(args, encoding="utf-8", env=env,
                                capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        error_msg = "Timed out after {} seconds".format(timeout_seconds)
        return QueryInfo(model_meta=model_meta, stdout="", stderr=error_msg)
    except Exception as e:
        print("Unknown error:", e)
        return QueryInfo(model_meta=model_meta, stdout="", stderr="Unknown error")

    return QueryInfo(model_meta=model_meta, stdout=output.stdout.strip(), stderr=output.stderr.strip())


def remove_models_curdir():
    for file in os.listdir():
        full_path = pathlib.Path(file).absolute()
        suffix = pathlib.Path(file).suffix
        if (suffix == ".xml") or (suffix == ".bin"):
            os.remove(full_path)


def query_proc_func(args):
    query_model, model_path, device, output_dir = args
    model_meta = utils.metadata_from_path(model_path)
    relative_dir = model_meta.model_path_relative
    utils.relative_change_dir(output_dir, relative_dir)
    query_info = one_model(
        query_model, model_path, device, model_meta)
    remove_models_curdir()

    return query_info


def all_models(query_model, models_list, device, output_dir):
    query_input = [(query_model, model_path, device, output_dir)
                   for model_path in models_list]
    return process_map(query_proc_func, query_input,
                       max_workers=os.cpu_count(), chunksize=1)
