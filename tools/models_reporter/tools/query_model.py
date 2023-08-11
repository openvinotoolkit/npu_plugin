#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
from pathlib import Path

import utils
from tools import parallel


def add_args(parser):
    group = parser.add_argument_group('Arguments for Query Model')
    group.add_argument(
        '-q', '--query_model',
        help='run query_model to collect unsupported layers',
        action='store_true',
        default=False
    )


class QueryModel(parallel.ProcessFunctions):
    def get_environment(self, binaries):
        env = os.environ.copy()
        env["VPUX_SERIALIZE_CANONICAL_MODEL"] = "1"
        env["IE_VPUX_LOG_LEVEL"] = 'LOG_ERROR'
        env["IE_VPUX_CREATE_EXECUTOR"] = '0'
        return env

    def get_argument_list(self, parser, plugin_config_path, model_path):
        query_model = parser.binaries / "query_model"
        return [query_model, "-m", model_path, "-d", parser.device]

    def cleanup(self, parser):
        [os.remove(file) for file in utils.glob_files(Path(), "*_canonical.xml")]
        [os.remove(file) for file in utils.glob_files(Path(), "*_canonical.bin")]


def all_models(parser, models_list, plugin_config_path):
    timeout = 10
    max_workers = parser.max_workers
    return parallel.all_models(parser, models_list, plugin_config_path, QueryModel(), timeout, max_workers)
