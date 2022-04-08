#!/usr/bin/python
#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from datetime import datetime, timezone, timedelta
import os
import argparse
import pathlib
import re
from pygit2 import Repository
import shutil

import utils
import layers
import compiler
import inference
import query
import dump
import table
import parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='models_path', type=pathlib.Path, required=True,
                        help='Path to models directory')
    parser.add_argument('-c', dest='compile_tool', type=pathlib.Path, required=True,
                        help='Path to compile_tool executable')
    parser.add_argument('-q', dest='query_model', type=pathlib.Path, required=False,
                        help='Path to query_model executable')
    parser.add_argument('-b', dest='benchmark_app', type=pathlib.Path, required=False,
                        help='Path to benchmark_app executable')
    parser.add_argument('-d', dest='device', default='VPUX.3720',
                        help='Device name')
    parser.add_argument('-t', dest='timeout', type=int, default='180',
                        help='Time cap for a compilation in seconds')
    parser.add_argument('-o', dest='output_path', type=pathlib.Path, default='blobs',
                        help='Output file')
    parser.add_argument('-r', dest='report_name', type=str,
                        help='Report file name')
    parser.add_argument('-ip', dest='input_precision', type=str, default='u8',
                        help='Input precision parameter for compile_tool')
    parser.add_argument('-op', dest='output_precision', type=str, default='fp16',
                        help='Output precision parameter for compile_tool')
    parser.add_argument('--disable-stubs', dest='disable_stubs', default=False, action='store_true',
                        help="Skip compilation with stubs that replaces unsupported SW kernels with dummy layers")
    parser.add_argument('--generate-layers-params', dest='generate_layers_params', default=False, action='store_true',
                        help='Insert LayerInModel table with layer parameters for each model')
    parser.add_argument('--use-irs', dest='use_irs', default=False, action='store_true',
                        help='Use IRs instead of blobs for the inference')

    return parser.parse_args()


def check_args(p):
    if "VPUX" not in p.device and not p.disable_stubs:
        print(f"Disable stubs for {p.device} device")
        p.disable_stubs = True

    if "VPUX" not in p.device and not p.use_irs:
        print(f"Use IRs instead of blobs for {p.device} device")
        p.use_irs = True

    if p.report_name and not p.report_name.endswith(".xlsx"):
        p.report_name = pathlib.Path(p.report_name).name.with_suffix(".xlsx")


def print_setup_config(c):
    print(f"OpenVINO:   {c.ov_branch} ({c.ov_commit})")
    print(f"VPUXPlugin: {c.vpux_branch} ({c.vpux_commit})")
    print(f"Commit date: {c.vpux_commit_date}")
    print("\nModel drops:")
    for drop in c.model_drops:
        print(drop)
    print()


def get_model_drops(models_list):
    drops_set = set(utils.metadata_from_path(
        model_path).package_name for model_path in models_list)
    return sorted(list(drops_set))


class SetupConfig:
    ov_branch: str
    ov_commit: str
    vpux_branch: str
    vpux_commit: str
    vpux_commit_date: datetime
    model_drops: list


def get_last_commit_time(repo):
    commit = repo.revparse_single("HEAD")
    tzinfo = timezone(timedelta(minutes=commit.author.offset))
    dt = datetime.fromtimestamp(float(commit.author.time), tzinfo)
    return dt.astimezone(tzinfo.utc).replace(tzinfo=None)


def get_setup_config(compile_tool, models_list):
    c = SetupConfig()

    try:
        ov_repo = Repository(compile_tool)
        c.ov_branch = ov_repo.head.shorthand
        c.ov_commit = ov_repo.head.target
    except:
        # TODO We not always use a repository
        c.ov_branch = "Not found"
        c.ov_commit = "Not found"

    try:
        ci_tag = re.compile("ci_tag_vpux_nightly_(\d+).*")
        binary_dir = str(pathlib.Path(compile_tool).parents[1].name)
        match = ci_tag.match(binary_dir)
        if match:
            c.vpux_branch = "Not found"
            c.vpux_commit = binary_dir
            date_str = match.groups()[0]
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            c.vpux_commit_date = datetime(year, month, day)
        else:
            vpux_repo = Repository(os.curdir)
            c.vpux_branch = vpux_repo.head.shorthand
            c.vpux_commit = vpux_repo.head.target
            c.vpux_commit_date = get_last_commit_time(vpux_repo)
    except:
        c.vpux_branch = "Not found"
        c.vpux_commit = "Not found"
        c.vpux_commit_date = "Not found"

    c.model_drops = get_model_drops(models_list)
    return c


def get_plugin_config(insert_stubs):
    config_list = []

    if insert_stubs:
        stub_key = "true" if insert_stubs else "false"
        config_list.append(
            f"VPUX_COMPILATION_MODE_PARAMS dummy-op-replacement={stub_key}")

    return config_list


def print_plugin_config(config_list):
    print("Plugin config:")
    for config in config_list:
        print(config)
    print()


def write_plugin_config_to_file(config_path, config_list):
    with open(config_path, "w") as f:
        f.writelines(config_list)

    return config_path


def glob_models_xml(models_path):
    models = []
    for root, dirs, files in os.walk(models_path, followlinks=True):
        for file in files:
            if file.endswith(".xml") and not file.endswith("manifest.xml"):
                model_path = pathlib.Path(os.path.join(root, file))
                models.append(model_path)
    return models


def get_models(models_path):
    models_list = glob_models_xml(models_path)
    print("Found {0} models".format(len(models_list)))

    remove_models = [
        model_path for model_path in models_list if model_path.name == 'intermediate_model.xml']
    if len(remove_models) > 0:
        print("Exclude */intermediate_model.xml models")
        models_list = [
            model_path for model_path in models_list if model_path not in remove_models]
        print("Remained {0} models".format(len(models_list)))

    return models_list


def append_tables_from_template(excel_table, benchmark_app):
    stats_df, compile_problems_df, infer_problems_df, tracking_df = dump.read_excel_template()
    excel_table.append_df("Statistics", stats_df, format=False, header=False)
    excel_table.append_df("CompilationProblems", compile_problems_df,
                          format=False, header=False)
    excel_table.append_df("ErrorTracking", tracking_df,
                          format=False, header=False)
    if benchmark_app:
        excel_table.append_df("InferenceProblems", infer_problems_df,
                              format=False, header=False)


def append_models_and_packages(excel_table, models_list):
    models_meta = [utils.metadata_from_path(path) for path in models_list]
    model_table, package_table, model_in_package_table = table.model_and_package(
        models_meta)
    excel_table.append(model_table)
    excel_table.append(package_table)
    excel_table.append(model_in_package_table)


def plugin_config_path(output_dir):
    return str(os.path.join(output_dir, "vpux.config"))


def append_compile_info(excel_table, compile_tool, models_list, device, input_precision, output_precision, output_dir, timeout, use_irs, config_path, report_id):
    print("Compilation...")
    compile_list = compiler.all_models(
        compile_tool, models_list, device, input_precision, output_precision, output_dir, timeout, use_irs, config_path)
    compilation_table = table.compilation(compile_list, report_id)
    excel_table.append(compilation_table)
    return compile_list


def append_config(excel_table, models_list, device, compile_tool, insert_stubs, config_path, report_id):
    setup_config = get_setup_config(compile_tool, models_list)
    print_setup_config(setup_config)

    plugin_config = get_plugin_config(insert_stubs)
    write_plugin_config_to_file(config_path, plugin_config)
    print_plugin_config(plugin_config)

    report_table = table.report(
        device, setup_config, plugin_config, report_id)
    excel_table.append(report_table)


def query_model(query_model, models_list, device, output_dir):
    query_list = []
    if query_model:
        print("Query_model...")
        query_list = query.all_models(
            query_model, models_list, device, output_dir)

    return query_list


def append_unsupported_layers_table(excel_table, models_meta, unsupported_models_layers):
    unsupported_ops_table = table.unsupported_layer(
        models_meta, unsupported_models_layers)
    excel_table.append(unsupported_ops_table)


def append_all_layers_table(excel_table, model_layers_list):
    all_layers_set = parse.get_all_layers_set(model_layers_list)
    layer_table = table.layer(all_layers_set)
    excel_table.append(layer_table)


def append_layers_params(excel_table, models_meta, model_layers_list):
    layer_in_model_table = table.layer_in_model(
        model_layers_list, models_meta)
    excel_table.append(layer_in_model_table)


def model_path_to_output_blob(model_path, output_dir):
    meta = utils.metadata_from_path(model_path)
    relative_path = meta.model_path_relative.with_suffix(".blob")
    return os.path.join(output_dir, relative_path)


def append_inference_info(excel_table, benchmark_app, models_list, device, output_dir, timeout, config_path, use_irs, report_id):
    print("Inference...")
    if not use_irs:
        models_list = [model_path_to_output_blob(
            model_path, output_dir) for model_path in models_list]

    inference_list = inference.all_models(
        benchmark_app, models_list, device, output_dir, timeout, config_path)

    inference_table = table.inference(inference_list, report_id)
    excel_table.append(inference_table)


def main():
    p = parse_args()
    check_args(p)
    models_list = get_models(p.models_path)

    output_dir = pathlib.Path(p.output_path).absolute()
    output_dir.mkdir(exist_ok=True)

    excel_table = dump.ExcelReport()

    append_tables_from_template(excel_table, p.benchmark_app)
    append_models_and_packages(excel_table, models_list)
    config_path = plugin_config_path(output_dir)

    model_layers_list = layers.glob_all_layers(models_list)
    append_all_layers_table(excel_table, model_layers_list)

    models_meta = [utils.metadata_from_path(path) for path in models_list]
    if p.generate_layers_params:
        append_layers_params(excel_table, models_meta, model_layers_list)

    query_list = query_model(p.query_model, models_list, p.device, output_dir)
    unsupported_models_layers = parse.unsupported_from_query_model(
        query_list, model_layers_list)
    append_unsupported_layers_table(
        excel_table, models_meta, unsupported_models_layers)

    def run_pipeline(models_list, insert_stubs, previous_compile_list=[]):
        report_id = datetime.now()
        append_config(excel_table, models_list, p.device,
                      p.compile_tool, insert_stubs, config_path, report_id)

        print("Compilation...")
        compile_list = compiler.all_models(
            p.compile_tool, models_list, p.device, p.input_precision, p.output_precision, output_dir, p.timeout, p.use_irs, config_path)

        if insert_stubs:
            def get_data(c): return (c.model_meta, c.stderr)
            prev_data = [get_data(c) for c in previous_compile_list]
            compile_list = [
                c for c in compile_list if get_data(c) not in prev_data]
            print(
                f"Additional errors from compilation with STUBs: {len(compile_list)}")

        compilation_table = table.compilation(compile_list, report_id)
        excel_table.append(compilation_table)
        unsupported = parse.unsupported_from_compile_info(
            compile_list, model_layers_list)
        append_unsupported_layers_table(excel_table, models_meta, unsupported)

        if p.benchmark_app is not None:
            compiled_models = [model for model, compile_info in zip(
                models_list, compile_list) if compile_info.compiled]
            append_inference_info(excel_table, p.benchmark_app, compiled_models, p.device,
                                  output_dir, p.timeout, config_path, p.use_irs, report_id)

        return compile_list

    compile_list = run_pipeline(models_list, insert_stubs=False)
    if not p.disable_stubs:
        not_compiled_models = [model for model, compile_info in zip(
            models_list, compile_list) if not compile_info.compiled and "Timed out" not in compile_info.stderr]
        run_pipeline(not_compiled_models, insert_stubs=True,
                     previous_compile_list=compile_list)

    report_path = p.report_name if p.report_name else dump.report_filename(
        p.device)
    dump.to_excel(excel_table, report_path)

    shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
