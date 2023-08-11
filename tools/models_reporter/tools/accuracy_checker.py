#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import copy
from dataclasses import dataclass
from utils import StrictTypeCheck
from functools import partial
import os
from pathlib import Path
import pathlib
import subprocess
import utils
from time import time
import pandas as pd
from multiprocessing import Pool

from tools import parallel


def add_args(parser):
    group = parser.add_argument_group('Arguments for the Accuracy Checker')
    group.add_argument(
        '-a', '--accuracy_checker',
        help='run accuracy_checker to collect compilation info',
        action='store_true',
        default=False
    )
    group.add_argument(
        '--accuracy_checker_src',
        help='path to Accuracy Checker directory with setup.py file',
        type=partial(utils.get_path, is_directory=True),
        required=False
    )
    group.add_argument(
        '--definitions',
        help='path to the yml file with definitions',
        type=partial(utils.get_path, is_directory=False),
        required=False
    )
    group.add_argument(
        '--configs',
        help='path to the directory with yml configurations',
        type=partial(utils.get_path, is_directory=True),
        required=False
    )
    group.add_argument(
        '--source',
        help='path to the validation dataset',
        type=partial(utils.get_path, is_directory=True),
        required=False
    )
    group.add_argument(
        '--annotation_converters_ext',
        help='path to the accuracy checker annotation converters extensions',
        type=partial(utils.get_path, is_directory=True),
        required=False
    )
    group.add_argument(
        '--pyenv',
        help='path to the python virtual environment',
        type=partial(utils.get_path, is_directory=True),
        default=Path("pyenv").absolute(),
        required=False
    )
    group.add_argument(
        '--annotations',
        help='path to the annotation conversion output directory',
        type=partial(utils.get_path, is_directory=True),
        default=Path("annotations").absolute(),
        required=False
    )
    group.add_argument(
        '--subsample_size',
        help="dataset subsample size",
        type=int,
        default=100,
        required=False
    )
    group.add_argument(
        '--accuracy_checker_timeout',
        help='time cap in seconds for accuracy_checker for one inference',
        type=int,
        default=10
    )


@dataclass
class Metric(StrictTypeCheck):
    name: str
    value: float
    cpu_value: float
    ground_truth: float
    metric_hint: str
    dataset_size: int


class AccuracyChecker(parallel.ProcessFunctions):
    def setup_env_and_validate(self, parser):
        parser.annotations.mkdir(exist_ok=True)

        if not os.path.exists(parser.pyenv):
            setup_python_env(parser.pyenv, parser.binaries, parser.accuracy_checker_src,
                             parser.annotation_converters_ext)
        # check to see if everything is ok
        self.get_environment(parser.binaries)

    def __init__(self, parser, configs):
        self.csv_result = utils.safe_to_save_filename(f"{parser.device}_{int(time())}.csv")
        self.models_with_config = copy.deepcopy(configs)
        self.setup_env_and_validate(parser)

    @staticmethod
    def find_python_dependencies_dir(binaries: Path):
        if os.name == 'nt':
            python_api = binaries / "Release" / "python_api"
        else:
            python_api = binaries / "python_api"

        if not python_api.is_dir():
            raise NotADirectoryError(python_api)

        subfolder = python_api.glob("python*")
        python_path = next(subfolder, None)
        if python_path is None:
            raise NotADirectoryError(f"{python_api} should contain python[version] inside")

        return python_path

    def get_environment(self, binaries):
        env = os.environ.copy()

        if os.name == 'nt':
            env["PATH"] += ";".join(
                [str(binaries), str(binaries / ".." / "deps")])
        else:
            env["LD_LIBRARY_PATH"] = ":".join(
                [str(binaries), str(binaries / "deps")])

        env["IE_VPUX_CREATE_EXECUTOR"] = "1"
        env["PYTHONPATH"] = str(self.find_python_dependencies_dir(binaries))
        env["OPENVINO_LIB_PATHS"] = str(binaries)

        return env

    def get_argument_list(self, parser, plugin_config_path, model_path):
        if os.path.exists(self.csv_result):
            os.remove(self.csv_result)

        if os.name == 'nt':
            accuracy_check = parser.pyenv / "Scripts" / "accuracy_check.exe"
        else:
            accuracy_check = parser.pyenv / "bin" / "accuracy_check"

        if model_path in self.models_with_config:
            config = self.models_with_config[model_path]
        else:
            config = Path("not_found.yml")

        args = [accuracy_check,
                "--definitions", parser.definitions,
                "--models", model_path,
                "--config", config,
                "--source", parser.source,
                "--annotations", parser.annotations,
                "--target_framework", "openvino",
                "--ie_preprocessing", "False",
                "--target_devices", parser.device,
                "--subsample_size", parser.subsample_size,
                "--use_new_api", "True",
                "--ignore_result_formatting", "True",
                "--shuffle", "False",
                "--csv_result", self.csv_result]

        if plugin_config_path:
            args.extend(["--device_config", plugin_config_path])

        return args

    def cleanup(self, parser):
        pass


def execute(command_list: list, capture_output=True):
    if os.name == 'nt':
        command_list = ["powershell", "-Command"] + command_list
    command_list = list(map(str, command_list))

    print(f"\n$ {' '.join(command_list)}")

    try:
        result = subprocess.run(
            command_list, encoding="utf-8", capture_output=capture_output)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        exit(1)

    if not capture_output:
        return ""

    output = result.stdout.strip()
    print(output)
    return output


def get_virtualenv_python_path(pyenv: list):
    if os.name == 'nt':
        return pyenv / "Scripts" / "python.exe"
    else:
        return pyenv / "bin" / "python3"


def setup_python_env(pyenv, binaries, accuracy_checker_dir, annotation_converters_ext):
    print("setup python virtual environment...")
    python = "python" if os.name == 'nt' else "python3"
    execute([python, "-m", "virtualenv", pyenv])
    python = get_virtualenv_python_path(pyenv)

    py_requirements = AccuracyChecker.find_python_dependencies_dir(binaries) / "requirements.txt"
    execute([python, "-m", "pip", "install", "-r", py_requirements])

    if annotation_converters_ext:
        execute([python, Path(annotation_converters_ext) / "setup.py",
                "install_as_extension", f"--accuracy-checker-dir={accuracy_checker_dir}"])

    execute([python, "-m", "pip", "install",
            accuracy_checker_dir / ".[extra]"], capture_output=False)


def find_configs(models_list, configs, get_config_name):
    models_with_config = []
    models_without_config = []

    models_meta = utils.get_models_meta(models_list)
    for model_meta in models_meta:
        model_config_name = get_config_name(model_meta)
        for config in configs:
            if config.name == model_config_name:
                models_with_config.append((model_meta.model_path, config))
                break
        else:
            models_without_config.append(model_meta.model_path)

    return models_with_config, models_without_config


def read_csv_metrics(csv_result):
    metrics = []
    try:
        if os.path.exists(csv_result):
            for line in pd.read_csv(csv_result, encoding='utf-8', chunksize=1):
                metric_data = line.iloc[0]
                metric = Metric(
                    name=metric_data["metric_name"],
                    value=float(metric_data["metric_value"]),
                    ground_truth=float(metric_data["ref"]),
                    metric_hint=metric_data["metric_target"],
                    dataset_size=int(metric_data["dataset_size"]),
                    cpu_value=float()
                )
                metrics.append(metric)
    except:
        return []

    return metrics


def juxtapose_models_and_configs(models_list, configs_path):
    configs = utils.glob_files(configs_path, "*.yml")

    models_with_config, models_without_config = find_configs(
        models_list, configs, lambda m: f"{m.model_name}-{m.framework}.yml")

    additional_models_with_config, models_without_config = find_configs(
        models_without_config, configs, lambda m: f"{m.model_name}.yml")

    models_with_config.extend(additional_models_with_config)

    print("models with config:", len(models_with_config))
    print("models without config:", len(models_without_config))

    model_config_dict = dict()
    for model_path, config_path in models_with_config:
        model_config_dict[model_path] = config_path

    return model_config_dict


def read_csv_results(csv_result, output_dir, models_list):
    root = Path().absolute()

    models_metrics = []
    for model_path in models_list:
        parallel.change_work_dir(output_dir, model_path)
        metrics = read_csv_metrics(csv_result)
        models_metrics.append(metrics)

    os.chdir(str(root))
    return models_metrics


def all_models(parser, models_list, plugin_config_path):
    model_config_dict = juxtapose_models_and_configs(models_list, parser.configs)
    timeout = parser.accuracy_checker_timeout * parser.subsample_size

    # CPU background AC task to get the reference
    cpu_max_workers = 1
    pool = Pool(cpu_max_workers)
    cpu_parser = copy.deepcopy(parser)
    cpu_parser.device = "CPU"
    cpu_ac = AccuracyChecker(cpu_parser, model_config_dict)
    cpu_func = partial(parallel.process_func, cpu_parser, None, cpu_ac, timeout)
    cpu_subtask = pool.map_async(cpu_func, models_list)

    # main computation
    ac = AccuracyChecker(parser, model_config_dict)
    max_workers = 1
    accuracy_list = parallel.all_models(
        parser, models_list, plugin_config_path, ac, timeout, max_workers)
    models_metrics = read_csv_results(ac.csv_result, parser.output_dir, models_list)
    for accuracy, metrics in zip(accuracy_list, models_metrics):
        accuracy.metrics = metrics

    if not cpu_subtask.ready():
        print("waiting for the CPU plugin to calculate reference values")
    _ = cpu_subtask.get()

    cpu_models_metrics = read_csv_results(cpu_ac.csv_result, parser.output_dir, models_list)

    for accuracy, cpu_metrics in zip(accuracy_list, cpu_models_metrics):
        for metric, cpu_metric in zip(accuracy.metrics, cpu_metrics):
            metric.cpu_value = cpu_metric.value

    return accuracy_list
