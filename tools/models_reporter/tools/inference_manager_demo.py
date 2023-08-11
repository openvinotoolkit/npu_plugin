#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
import utils
import shutil
import multiprocessing

from tools import parallel
from functools import partial


def add_args(parser):
    group = parser.add_argument_group('Arguments for Inference Manager Demo')
    group.add_argument(
        '--imd',
        help='path to Inference Manager Demo application',
        type=partial(utils.get_path, is_directory=True),
    )
    group.add_argument(
        '--config_file',
        help='config file for Inference Manager Demo to use',
        type=str,
        default='.config_3720xx.perf'
    )
    group.add_argument(
        '--imd_timeout',
        help='time cap in seconds for Inference Manager Demo execution',
        type=int,
        default=180
    )
    group.add_argument(
        '--srv_ip',
        help='remote IP with debug server running',
        type=str
    )
    group.add_argument(
        '--srv_port',
        help='remote port to connect to',
        type=str
    )


def build_command_list(parser):
    args = ["make", f"CONFIG_FILE={parser.config_file}",
            f"-j{multiprocessing.cpu_count()}", "--directory", parser.imd]

    return args


def build(parser):
    args = build_command_list(parser)
    print(" ".join(map(str, args)))
    env = os.environ.copy()
    timeout = 180
    parallel.one_model(args, env, timeout)


class InferenceManagerDemo(parallel.ProcessFunctions):
    def get_environment(self, binaries):
        env = os.environ.copy()
        return env

    def get_argument_list(self, parser, plugin_config_path, model_path):
        shutil.copy(model_path, parser.imd / "test.blob")

        args = build_command_list(parser) + ["run"]

        if parser.srv_ip:
            args = args + [f"srvIP={parser.srv_ip}"]

        if parser.srv_port:
            args = args + [f"srvPort={parser.srv_port}"]

        return args

    def cleanup(self, parser):
        pass


def all_models(parser, models_list, plugin_config_path):
    timeout = parser.imd_timeout
    max_workers = 1
    return parallel.all_models(parser, models_list, plugin_config_path, InferenceManagerDemo(), timeout, max_workers)
