#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import partial
import os
import argparse
from pathlib import Path
import re
import shutil
from time import time
from pygit2 import Repository
from get_version import get_version_by_compilation
import shutil


from tools import compile_tool
from tools import query_model
from tools import benchmark_app
from tools import accuracy_checker
from tools import inference_manager_demo
from tools import single_image_test
from tools import bin_diff

import utils
import layers
import dump
import table
import parse

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


def add_args(parser):
    group = parser.add_argument_group('Common arguments for Model\'s reporter')
    group.add_argument(
        '-m', '--models', '--network',
        help='path to models directory',
        type=partial(utils.get_path, file_or_directory=True),
        required=True,
        nargs='+'
    )
    group.add_argument(
        '--binaries',
        help='path to the binaries directory',
        type=partial(utils.get_path, is_directory=True),
        required=True
    )
    group.add_argument(
        '-d', '--device',
        help='device name',
        type=str,
    )
    group.add_argument(
        '-r', '--report_path',
        help='path to the output/report.xlsx or just output/ directory',
        type=Path,
        default=Path("reports").absolute()
    )
    group.add_argument(
        '--output_dir',
        help='path to the output directory to store blobs and temporary files',
        type=partial(utils.get_path, is_directory=True, check_exists=False),
        default=Path("blobs").absolute()
    )
    group.add_argument(
        '--disable_stubs',
        help="disable second pipeline run with stubs that replaces unsupported SW kernels with dummy layers",
        action='store_true',
    )
    group.add_argument(
        '-l', '--layer_params',
        help='insert a table with all layers and their parameters in provided models list',
        action='store_true',
    )
    group.add_argument(
        '--use_irs',
        help='use IRs instead of blobs for the inference',
        action='store_true',
    )
    group.add_argument(
        '--black_list',
        help='path to the black list file. each line has regex that applies to the model\'s path',
        type=partial(utils.get_path, file_or_directory=True, is_directory=False)
    )
    group.add_argument(
        '--use_template',
        help='import models_report_template.xlsx to have it as a baseline for the report',
        action='store_true',
    )
    group.add_argument(
        '--max_workers',
        help='maximum number of workers (processes) to parallel some tasks',
        type=int,
        default=os.cpu_count()
    )
    group.add_argument(
        '-ip', '--input_precision',
        help='input precision parameter',
        type=str,
        default='FP16'
    )
    group.add_argument(
        '-op', '--output_precision',
        help='output precision parameter',
        type=str,
        default='FP16'
    )


class SetupConfig:
    ov_revision: str
    vpux_revision: str
    vpux_commit_date: datetime
    model_drops: list


def get_last_commit_time(repo):
    commit = repo.revparse_single("HEAD")
    tzinfo = timezone(timedelta(minutes=commit.author.offset))
    dt = datetime.fromtimestamp(float(commit.author.time), tzinfo)
    return dt.astimezone(tzinfo.utc).replace(tzinfo=None)


def get_model_drops(models_list):
    drops_set = set(utils.metadata_from_path(model_path).package_name for model_path in models_list)
    return sorted(list(drops_set))


def get_setup_config(binaries, output_dir, models_list):
    c = SetupConfig()
    c.ov_revision, c.vpux_revision = get_version_by_compilation(binaries, output_dir)

    package_dir = binaries.parent.name if os.name == 'nt' else binaries.name
    match = re.search("^(\d{4})(\d{2})(\d{2}).*", package_dir)
    if match:
        year = int(match.groups()[0])
        month = int(match.groups()[1])
        day = int(match.groups()[2])
        c.vpux_commit_date = datetime(year, month, day)
    else:
        try:
            vpux_repo = Repository(os.curdir)
            c.vpux_commit_date = get_last_commit_time(vpux_repo)
        except:
            c.vpux_commit_date = "Not found"

    c.model_drops = get_model_drops(models_list)
    return c


def get_plugin_config(device, insert_stubs, dpu_groups):
    config_list = []

    if insert_stubs:
        stub_key = "true" if insert_stubs else "false"
        config_list.append(f"VPUX_COMPILATION_MODE_PARAMS dummy-op-replacement={stub_key}")

    if device == "VPUX.4000":
        dpu_groups = 1
        print(f"Set VPUX_DPU_GROUPS to {dpu_groups} for {device}")

    if dpu_groups:
        config_list.append(f"VPUX_DPU_GROUPS={dpu_groups}")

    if device.find("HETERO") != -1:
        searchResult = re.match(".*VPUX.([0-9]+).*", device)
        if searchResult is None:
            print("Incorrect platform was provided for HETERO. Example:HETERO:VPUX.3720,CPU")
        else:
            platform = "VPUX_PLATFORM " + searchResult.group(1)
            config_list.append(platform)

    return config_list


def print_plugin_config(config_list):
    if config_list:
        print("\n\t".join(["Plugin config:"] + config_list))


def write_plugin_config_to_file(plugin_config_path, config_list):
    with open(plugin_config_path, "w") as f:
        f.writelines(line + '\n' for line in config_list)
    return plugin_config_path


def get_models_from_paths(models_sources: list):
    models_list = utils.flatten(utils.glob_files(models_path, "*.xml")
                                for models_path in models_sources)
    models_list = [m for m in models_list if m.name != "manifest.xml"]

    print(f"Found {len(models_list)} models")

    models_to_remove = [
        model_path for model_path in models_list if model_path.name == 'intermediate_model.xml']
    if len(models_to_remove) > 0:
        print("Exclude */intermediate_model.xml models")
        models_list = [model_path for model_path in models_list if model_path not in models_to_remove]
        print(f"Remained {len(models_list)} models")

    return models_list


def filter_using_black_list(models_list, black_list_path):
    with open(black_list_path) as f:
        for line in f.readlines():
            models_count = len(models_list)
            line = line.strip()
            regex = re.compile(line)
            [path for path in models_list if regex.match(str(path))]
            models_list = [path for path in models_list if not regex.match(str(path))]
            removed = models_count - len(models_list)
            print(f"Removed {removed} models using regex: {line}")

    print(f"Remained {len(models_list)} models")
    return models_list


def append_tables_from_template(excel_table, benchmark_app):
    stats_df, compile_problems_df, infer_problems_df, tracking_df = dump.read_excel_template()
    excel_table.append_df("Statistics", stats_df, format=False, header=False)
    excel_table.append_df("CompilationProblems", compile_problems_df, format=False, header=False)
    excel_table.append_df("ErrorTracking", tracking_df, format=False, header=False)
    if benchmark_app:
        excel_table.append_df("InferenceProblems", infer_problems_df, format=False, header=False)


def append_models_and_packages(excel_table, models_list):
    models_meta = utils.get_models_meta(models_list)
    model_table, package_table, model_in_package_table = table.model_and_package(models_meta)
    excel_table.append(model_table)
    excel_table.append(package_table)
    excel_table.append(model_in_package_table)


def get_plugin_config_path(output_dir: Path):
    return output_dir / "vpux.config"


def print_setup_config(c):
    print(f'{"OpenVINO: ":13}{c.ov_revision}')
    print(f'{"VPUXPlugin: ":13}{c.vpux_revision}')
    print(f'{"Commit date: ":13}{c.vpux_commit_date}')
    print("\n\t".join(["Model drops:"] + c.model_drops))


def append_config(excel_table, models_list, parser, insert_stubs, plugin_config_path, report_id):
    setup_config = get_setup_config(parser.binaries, parser.output_dir, models_list)
    print_setup_config(setup_config)

    plugin_config = get_plugin_config(parser.device, insert_stubs, parser.dpu_groups)
    write_plugin_config_to_file(plugin_config_path, plugin_config)
    print_plugin_config(plugin_config)

    report_table = table.report(parser.device, setup_config, plugin_config, report_id)
    excel_table.append(report_table)


def append_unsupported_layers_table(excel_table, models_list, unsupported_models_layers):
    models_meta = utils.get_models_meta(models_list)
    unsupported_ops_table = table.unsupported_layer(models_meta, unsupported_models_layers)
    excel_table.append(unsupported_ops_table)


def append_all_layers_table(excel_table, model_layers_list):
    all_layers_set = parse.get_all_layers_set(model_layers_list)
    layer_table = table.layer(all_layers_set)
    excel_table.append(layer_table)


def append_layers_params(excel_table, models_list, model_layers_list, max_workers):
    models_meta = utils.get_models_meta(models_list)
    layer_in_model_table = table.layer_in_model(model_layers_list, models_meta, max_workers)
    excel_table.append(layer_in_model_table)


def select_successful(models_list, result_list):
    return [model for model, res in zip(models_list, result_list) if res.return_code == 0]


def warn(message):
    print(f"{Fore.YELLOW}WARN: {message}{Style.RESET_ALL}")


class ModelForm(Enum):
    BLOB = 0
    IR = 1


def models_or_blobs(models_list: Path, output_dir: Path, model_form: ModelForm):
    return models_list if model_form == ModelForm.IR else [utils.model_to_blob_path(model_path, output_dir) for model_path in models_list]


def select_new_errors(compile_list, models_list, models_error):
    new_compile_list = [compile_info for compile_info, model_path in zip(
        compile_list, models_list) if models_error[model_path] != compile_info.stderr]
    new_models_list = [model_path for compile_info, model_path in zip(
        compile_list, models_list) if models_error[model_path] != compile_info.stderr]

    return new_compile_list, new_models_list


def run_pipeline(excel_table, parser, plugin_config_path, models_list, model_layers_list, insert_stubs, models_error=[]):
    report_id = datetime.now()

    if parser.compile_tool:
        remove_output_dir(parser.output_dir)

    parser.output_dir.mkdir(exist_ok=True)
    append_config(excel_table, models_list, parser, insert_stubs, plugin_config_path, report_id)

    compile_list = []
    if parser.compile_tool:
        print("compile_tool...")
        compile_list = compile_tool.all_models(parser, models_list, plugin_config_path)

        if insert_stubs:
            compile_list, models_list = select_new_errors(compile_list, models_list, models_error)
            print(f"models with different error when compiled with STUBs: {len(compile_list)}")

        models_meta = utils.get_models_meta(models_list)
        has_conformance = any('conformance' in utils.get_source(model) for model in models_meta)
        if has_conformance:
            conformance_table = table.conformance(
                compile_list, models_list, parser, plugin_config_path)
            excel_table.append(conformance_table)
        else:
            compilation_table = table.compilation(
                compile_list, models_list, report_id, parser, plugin_config_path)
            excel_table.append(compilation_table)

        unsupported = parse.unsupported_from_compile_info(compile_list, model_layers_list)
        append_unsupported_layers_table(excel_table, models_list, unsupported)

        models_list = select_successful(models_list, compile_list)

    single_image_list_CPU = []
    single_image_list_VPU = []
    if parser.single_image_test:
        print("single-image-test...")
        for m in models_list:
            N_in, C_in, H_in, W_in = utils.get_input_shape_model(m)
            utils.generate_random_gradient_image(C_in, H_in, W_in, m)

        parser.device = "CPU"
        print("Run on CPU for SIT")
        single_image_list_CPU = single_image_test.all_models(
            parser, models_list, plugin_config_path)
        utils.append_device_to_blob_name(parser.output_dir, "_CPU")

        parser.device = "CPU"
        print("Run on VPU for SIT")
        single_image_list_VPU = single_image_test.all_models(
            parser, models_list, plugin_config_path)
        utils.append_device_to_blob_name(parser.output_dir, "_VPU")

    if parser.benchmark_app:
        model_form = ModelForm.IR if parser.use_irs else ModelForm.BLOB
        models_or_blobs_list = models_or_blobs(models_list, parser.output_dir, model_form)

        print("benchmark_app...")
        inference_list = benchmark_app.all_models(parser, models_or_blobs_list, plugin_config_path)

        inference_table = table.inference(inference_list, models_list, report_id)
        excel_table.append(inference_table)

        models_list = select_successful(models_list, inference_list)

    if parser.imd and not insert_stubs:
        model_form = ModelForm.IR if parser.use_irs else ModelForm.BLOB
        if model_form == ModelForm.IR:
            warn("Can't run IMD with IRs. Skipping")
        else:
            models_or_blobs_list = models_or_blobs(models_list, parser.output_dir, model_form)

            print("build inference manager demo:")
            inference_manager_demo.build(parser)
            print("inference_manager_demo...")
            imd_list = inference_manager_demo.all_models(
                parser, models_or_blobs_list, plugin_config_path)

            imd_table = table.inference_manager_demo(imd_list, models_list, report_id, parser)
            excel_table.append(imd_table)

    if parser.accuracy_checker and not insert_stubs:
        print("accuracy_checker...")
        model_form = ModelForm.IR if parser.use_irs else ModelForm.BLOB
        if not parser.use_irs:
            # force use IRs
            model_form = ModelForm.IR
            warn("accuracy_check returns 0xC0000005 error when blobs are provided. Falling back to using IRs.")

        models_or_blobs_list = models_or_blobs(
            models_list, parser.output_dir, model_form=model_form)

        accuracy_list = accuracy_checker.all_models(parser, models_list, plugin_config_path)
        models_meta = utils.get_models_meta(models_list)
        accuracy_table = table.accuracy(accuracy_list, models_meta, report_id)
        excel_table.append(accuracy_table)

    return compile_list


def get_models_error(compile_list, models_list):
    models_error = dict()
    for compile_info, model_path in zip(compile_list, models_list):
        if compile_info.return_code != 0 and "Timed out" not in compile_info.stderr:
            models_error[model_path] = compile_info.stderr

    return models_error


def check_args(parser):
    if parser.device.find("HETERO") != -1:
        if not parser.disable_stubs:
            warn(f"Disable stubs for HETERO plugin")
            parser.disable_stubs = True
        if not parser.use_irs:
            warn(f"Use IRs for the HETERO plugin")
            parser.use_irs = True

    if "VPUX" not in parser.device and not parser.disable_stubs:
        warn(f"Disable stubs for {parser.device} device")
        parser.disable_stubs = True

    if not parser.compile_tool:
        warn(f"Compile Tool is disabled, use IRs for inference")
        parser.use_irs = True

    if parser.imd:
        if not parser.srv_ip:
            warn("--srv_ip parameter is not set, will be using local server")

    parser.report_path = parser.report_path.absolute()
    parser.output_dir = parser.output_dir.absolute()
    parser.binaries = parser.binaries.absolute()

    def executable_exists(filename):
        path = parser.binaries / filename

        if os.name == 'nt':
            path = path.with_suffix(".exe")

        if not os.path.exists(str(path)):
            raise FileNotFoundError(f"File not found: {path}")

    if parser.compile_tool:
        executable_exists("compile_tool")

    if parser.benchmark_app:
        executable_exists("benchmark_app")

    if parser.query_model:
        executable_exists("query_model")

    if parser.accuracy_checker:
        accuracy_checker.AccuracyChecker(parser, configs=None)

    if parser.single_image_test:
        executable_exists("single-image-test")


def parse_args():
    parser = argparse.ArgumentParser()

    add_args(parser)
    compile_tool.add_args(parser)
    query_model.add_args(parser)
    benchmark_app.add_args(parser)
    accuracy_checker.add_args(parser)
    inference_manager_demo.add_args(parser)
    single_image_test.add_args(parser)

    return parser.parse_args()


def colored_status(boolean):
    if boolean:
        return f"{Fore.GREEN}ON{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}OFF{Style.RESET_ALL}"


def print_status(parser):
    print(f'{"compile_tool:":24}{colored_status(parser.compile_tool)}')
    print(f'{"query_model:":24}{colored_status(parser.query_model)}')
    print(f'{"benchmark_app:":24}{colored_status(parser.benchmark_app)}')
    print(f'{"inference_manager_demo:":24}{colored_status(parser.imd)}')
    print(f'{"single-image-test:":24}{colored_status(parser.single_image_test)}')
    print(f'{"accuracy_checker:":24}{colored_status(parser.accuracy_checker)}')
    print(f'{"use blobs:":24}{colored_status(not parser.use_irs)}')
    print(f'{"stub analysis:":24}{colored_status(not parser.disable_stubs)}')
    print(f'{"generate layer params:":24}{colored_status(parser.layer_params)}')


def configuration_abbreviation(parser):
    result = []
    if not parser.disable_stubs:
        result.append("S")
    if parser.layer_params:
        result.append("L")
    if parser.imd:
        result.append("I")
    if parser.accuracy_checker:
        result.append("A")
    if parser.benchmark_app:
        result.append("B")
    if parser.compile_tool:
        result.append("C")
    if parser.query_model:
        result.append("Q")
    if parser.single_image_test:
        result.append("SIT")
    return "".join(result)


def seconds_to_time(time_delta):
    hours = time_delta // 3600
    time_delta = time_delta % 3600
    minutes = time_delta // 60
    seconds = time_delta % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def remove_output_dir(output_dir):
    if os.path.exists(output_dir):
        # would love to use Path.is_relative_to, but Python3.9 is required
        safe_to_remove = str(output_dir).startswith(str(Path().absolute()))
        if safe_to_remove:
            print(f"Clearing output directory {output_dir}")
            shutil.rmtree(output_dir)


def main():
    script_start = time()

    parser = parse_args()
    print_status(parser)
    check_args(parser)
    models_list = get_models_from_paths(parser.models)

    if parser.black_list:
        models_list = filter_using_black_list(models_list, parser.black_list)

    excel_table = dump.ExcelReport()
    if parser.use_template:
        append_tables_from_template(excel_table, parser.benchmark_app)

    models_meta = utils.get_models_meta(models_list)
    has_conformance = any('conformance' in utils.get_source(model) for model in models_meta)
    if has_conformance:
        warn("Conformance model detected. Models and packages info will not be added to the report")
    else:
        append_models_and_packages(excel_table, models_list)

    print("glob all the layers...")
    model_layers_list = layers.glob_all_layers(models_list, parser.max_workers)
    append_all_layers_table(excel_table, model_layers_list)

    if parser.layer_params:
        append_layers_params(excel_table, models_list, model_layers_list, parser.max_workers)

    query_list = []
    if parser.query_model:
        print("query_model...")
        query_list = query_model.all_models(parser, models_list, plugin_config_path=None)

    unsupported_models_layers = parse.unsupported_from_query_model(query_list, model_layers_list)
    append_unsupported_layers_table(excel_table, models_list, unsupported_models_layers)

    plugin_config_path = get_plugin_config_path(parser.output_dir)

    compile_list = run_pipeline(excel_table, parser, plugin_config_path, models_list,
                                model_layers_list, insert_stubs=False)

    if not parser.disable_stubs:
        models_error = get_models_error(compile_list, models_list)
        if len(models_error) == 0:
            print("all models successfully compiled, nothing to run with STUBs")
        else:
            failed_models = models_error.keys()
            run_pipeline(excel_table, parser, plugin_config_path, failed_models,
                         model_layers_list, insert_stubs=True, models_error=models_error)

    if parser.single_image_test:
        cpu_filename, vpu_filename = utils.get_blob_file_paths(parser.output_dir)

        for cpu_file in cpu_filename:
            for vpu_file in vpu_filename:
                if utils.verify_matching_blobs(cpu_file, vpu_file):
                    xml_file, modelId = utils.find_matching_xml(models_list, cpu_file)
                    if utils.get_input_shape_model(xml_file):
                        values = utils.get_output_shape_model(xml_file)
                        if len(values) == 1 and isinstance(values[0], list):
                            N_out, C_out, H_out, W_out = 1, 1, 1, 1
                            if len(values[0]) == 2:
                                N_out, C_out = values[0]
                            elif len(values[0]) == 3:
                                C_out, H_out, W_out = values[0]
                            elif len(values[0]) == 4:
                                N_out, C_out, H_out, W_out = values[0]
                            metrics = bin_diff.compare_CPU_VPU(
                                cpu_file, vpu_file, N_out, C_out, H_out, W_out)
                            single_test_table = table.sit(modelId, metrics)
                            excel_table.append(single_test_table)

    if Path(parser.report_path).suffix == ".xlsx":
        report_path = parser.report_path
    else:
        report_path = parser.report_path / dump.report_filename(parser.device)

    abbreviation = configuration_abbreviation(parser)
    filename = f"{report_path.stem}_{abbreviation}{report_path.suffix}"
    report_path = report_path.with_name(filename)

    dump.to_excel(excel_table, report_path)

    time_delta = int(time() - script_start)
    print(f"script took {seconds_to_time(time_delta)} to run")


if __name__ == "__main__":
    main()
