#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
from pathlib import Path

import utils
from tools import parallel


def add_args(parser):
    group = parser.add_argument_group('Arguments for Benchmark App')
    group.add_argument(
        '-b', '--benchmark_app',
        help='run benchmark_app to collect inference info',
        action='store_true',
        default=False
    )
    group.add_argument(
        '--benchmark_app_timeout',
        help='time cap in seconds for benchmark_app',
        type=int,
        default=180
    )
    group.add_argument(
        '--benchmark_app_time_to_run',
        help='number of seconds to run benchmark_app for',
        type=int
    )


class BenchmarkApp(parallel.ProcessFunctions):
    def get_environment(self, binaries):
        env = os.environ.copy()
        env["IE_VPUX_LOG_LEVEL"] = 'LOG_ERROR'
        env["IE_VPUX_CREATE_EXECUTOR"] = "1"
        if os.name == 'nt':
            env["PATH"] += ";".join(
                [str(binaries), str(binaries / ".." / "deps")])
        else:
            env["LD_LIBRARY_PATH"] = ":".join(
                [str(binaries), str(binaries / "deps")])

        return env

    def get_argument_list(self, parser, plugin_config_path, model_path):
        # TODO plugin_config_path ignored for now since require .json instead of .txt
        # TODO add inference time measurements even for one inference. Right now it's FEIL
        nireq = 1
        benchmark_app = parser.binaries / "benchmark_app"
        args = [benchmark_app, "-d", parser.device, "-m",
                model_path, "-nireq", nireq, "--api", "sync"]

        if parser.benchmark_app_time_to_run:
            args.extend(["-t", parser.benchmark_app_time_to_run])
        else:
            args.extend(["-niter", 1])

        return args

    def cleanup(self, parser):
        pass


def all_models(parser, models_list, plugin_config_path):
    timeout = parser.benchmark_app_timeout
    max_workers = 1
    return parallel.all_models(parser, models_list, plugin_config_path, BenchmarkApp(), timeout, max_workers)
