import os
from pathlib import Path

import utils
from tools import parallel
from functools import partial
import numpy as np
import scipy
from scipy import special


def add_args(parser):
    group = parser.add_argument_group('Arguments for SIT')
    group.add_argument(
        '-sit', '--single_image_test',
        help='run single_image_test',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-tsit', '--timeout_single_image_test',
        help='time cap in seconds for single_image_test',
        type=int,
        default=180
    )
    parser.add_argument(
        '--input',
        help='Generate random input images for SIT',
        type=str,
        default='gradient',
        required=False
    )


class single_image_test(parallel.ProcessFunctions):
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
        single_image_test = parser.binaries / "single-image-test"
        if parser.input == 'gradient':
            path = os.path.dirname(model_path)
            parser.input = f'{path}/random_gradient_image.png'
        return [single_image_test, "--network", model_path, "--device", parser.device, "--input", parser.input, "-ip", parser.input_precision, "-op", parser.output_precision]

    def cleanup(self, parser):
        pass


def all_models(parser, models_list, plugin_config_path):
    timeout = parser.timeout_single_image_test
    max_workers = 4
    return parallel.all_models(parser, models_list, plugin_config_path, single_image_test(), timeout, max_workers)
