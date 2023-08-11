#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
from pathlib import Path

import utils
from tools import parallel


def add_args(parser):
    group = parser.add_argument_group('Arguments for Compile Tool')
    group.add_argument(
        '-c', '--compile_tool',
        help='run compile_tool to collect compilation info',
        action='store_true',
        default=False
    )
    group.add_argument(
        '--compile_tool_timeout',
        help='time cap in seconds for compile_tool for one model',
        type=int,
        default=180
    )
    group.add_argument(
        '--dpu_groups',
        help='number of dpu groups',
        type=int
    )


class CompileTool(parallel.ProcessFunctions):
    def get_environment(self, binaries):
        env = os.environ.copy()
        env["IE_VPUX_LOG_LEVEL"] = "LOG_ERROR"
        env["IE_VPUX_CREATE_EXECUTOR"] = "0"
        if os.name == 'nt':
            env["PATH"] += ";".join(
                [str(binaries), str(binaries / ".." / "deps")])
        else:
            env["LD_LIBRARY_PATH"] = ":".join(
                [str(binaries), str(binaries / "deps")])

        return env

    def get_argument_list(self, parser, plugin_config_path, model_path):
        compile_tool = parser.binaries / "compile_tool"
        return [compile_tool, "-d", parser.device, "-m", model_path,
                "-ip", parser.input_precision, "-op", parser.output_precision,
                "-c", plugin_config_path]

    def cleanup(self, parser):
        if parser.use_irs:
            [os.remove(file) for file in utils.glob_files(Path(), "*.blob")]


def all_models(parser, models_list, plugin_config_path):
    timeout = parser.compile_tool_timeout
    max_workers = parser.max_workers
    return parallel.all_models(parser, models_list, plugin_config_path, CompileTool(), timeout, max_workers)
