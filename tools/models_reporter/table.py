#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import numpy as np
import re
import pandas as pd
from itertools import groupby
from tqdm.contrib.concurrent import process_map
import utils
from tools.compile_tool import CompileTool


def chunks(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]


class DataTable:
    def __init__(self, name, header):
        self.name = name
        self.header = header
        self.data = []

    def append_row(self, row):
        self.data.extend(row)

    def get_table(self):
        width = len(self.header)
        if (len(self.data) % width != 0):
            raise Exception(
                f"Table `{self.name}` has row width not matching to number of columns")

        shape = (len(self.data) // width, width)
        table = np.array(self.data, dtype=object)
        table = np.reshape(table, newshape=shape)
        return table

    def remove_duplicates(self):
        unique_list = [key for key, _ in groupby(
            sorted(chunks(self.data, len(self.header))))]
        self.data = [item for row in unique_list for item in row]  # flatten
        return self

    def to_data_frame(self):
        table = self.get_table()
        df = pd.DataFrame(table, columns=self.header)

        column_headers = list(df.columns.values)
        return df


def report(device, setup_config, plugin_config_list, report_id):
    report_table = DataTable(
        "Report", ["device", "ovRevision", "vpuxRevision", "vpuxCommitDate", "pluginConfig", "reportId"])

    ov_revision = setup_config.ov_revision
    vpux_revision = setup_config.vpux_revision
    vpux_commit_date = setup_config.vpux_commit_date
    plugin_config = ",".join(plugin_config_list)

    row = [device, ov_revision, vpux_revision, vpux_commit_date, plugin_config, report_id]
    report_table.append_row(row)

    return report_table


def group_models_by_package(models_meta):
    package_models = dict()
    for package_name, model_meta in groupby(models_meta, lambda meta: meta.package_name):
        package_models[package_name] = list(model_meta)

    return package_models


def create_primary_key(values: list):
    return "-".join(map(str, values))


def get_model_id(model_meta):
    topology = model_meta.model_name
    framework = model_meta.framework
    precision = model_meta.precision
    batch = model_meta.batch
    return create_primary_key([topology, framework, precision, batch])


def get_package_id(model_meta):
    return model_meta.package_name


def get_model_in_package_id(model_meta):
    model_id = get_model_id(model_meta)
    package_id = get_package_id(model_meta)
    return create_primary_key([package_id, model_id])


def model_and_package(models_meta):
    package_table = DataTable(
        "Package", ["source", "packageType", "workWeek", "packageId"])

    model_table = DataTable(
        "Model", ["topology", "framework", "precision", "batch", "modelId"])

    model_in_package_table = DataTable(
        "ModelInPackage", ["packageId", "modelId", "modelInPackageId"])

    package_models_dict = group_models_by_package(models_meta)

    for package_name in package_models_dict:
        package_id = package_name
        source = utils.get_source(package_name)
        package_type = utils.get_package_type(package_name)
        work_week = utils.get_work_week(package_name)

        package_row = [source, package_type, work_week, package_id]
        package_table.append_row(package_row)

        for model_meta in package_models_dict[package_name]:
            topology = model_meta.model_name
            framework = model_meta.framework
            precision = model_meta.precision
            batch = int(model_meta.batch)

            model_row = [topology, framework, precision, batch]
            model_id = create_primary_key(model_row)
            model_table.append_row(model_row + [model_id])

            model_in_package_row = [package_id, model_id]
            model_in_package_id = create_primary_key(model_in_package_row)
            model_in_package_table.append_row(
                model_in_package_row + [model_in_package_id])

    # Models might have duplicates from different packages, remove
    model_table.remove_duplicates()

    return model_table, package_table, model_in_package_table


def layer(all_layers_set):
    layer_table = DataTable("Layer", ["name", "opset", "layerId"])

    for name, opset in sorted(list(all_layers_set)):
        row = [name, opset]
        layer_id = create_primary_key(row)
        layer_table.append_row(row + [layer_id])

    return layer_table


def sit(model_name: str, args):
    single_image_table = DataTable("SingleImageTest", [
                                   "modelId", "cosine", "MAE", "MSE", "Jaccard1", "Jaccard2", "correlation", "mean", "var"])

    metrics_list = np.array([args])

    for row in metrics_list:
        row_with_model_name = [model_name] + row.tolist()
        single_image_table.append_row(row_with_model_name)

    return single_image_table


def layers_in_model_func(args):
    model_layers, model_in_package_id = args
    layers_with_params = []
    for layer in model_layers:
        layer_id = create_primary_key([layer.name, layer.opset])
        input_str = ", ".join(str(shape) for shape in layer.input_shapes)
        output_str = ", ".join(str(shape) for shape in layer.output_shapes)

        row = [input_str, output_str,
               layer.attributes, model_in_package_id, layer_id]
        layers_with_params.append(row)

    return layers_with_params


def layer_in_model(model_layers_list, models_meta, max_workers):
    layer_in_model_table = DataTable(
        "LayerInModel", ["inputShapes", "outputShapes", "attributes", "modelInPackageId", "layerId"])

    ids = [get_model_in_package_id(model_meta) for model_meta in models_meta]

    input = [(layers, id) for layers, id in zip(model_layers_list, ids)]

    print("glob all layer parameters...")
    list_of_model_layers = process_map(
        layers_in_model_func, input, max_workers=max_workers, chunksize=1)

    for model_layers in list_of_model_layers:
        for layer in model_layers:
            layer_in_model_table.append_row(layer)

    return layer_in_model_table.remove_duplicates()


def unsupported_layer(models_meta, unsupported_models_layers):
    unsupported_layer_table = DataTable(
        "UnsupportedLayer", ["source", "package", "topology", "framework", "precision", "name", "opset", "reason", "layerId", "modelInPackageId"])

    has_conformance = any('conformance' in utils.get_source(model) for model in models_meta)
    for m, unsupported_model_layers in zip(models_meta, unsupported_models_layers):
        model_in_package_id = get_model_in_package_id(m)
        for name, opset, reason in unsupported_model_layers:
            layer_id = create_primary_key([name, opset])

            if has_conformance:
                new_row = [m.source, m.package_type, m.subgraph_name, m.framework,
                           m.subgraph_data_type, name, opset, reason, layer_id, model_in_package_id]
            else:
                new_row = [m.source, m.package_type, m.model_name, m.framework,
                           m.precision, name, opset, reason, layer_id, model_in_package_id]
            unsupported_layer_table.append_row(new_row)

    return unsupported_layer_table


def compilation(compile_list, models_list, report_id, parser, plugin_config_path):
    compilation_table = DataTable(
        "Compilation", ["source", "package", "topology", "framework", "precision", "status", "compileTimeMs", "errorId", "logs", "reportId", "modelInPackageId", "args"])

    models_meta = utils.get_models_meta(models_list)
    for c, m in zip(compile_list, models_meta):
        status = "PASSED" if c.return_code == 0 else "FAILED"
        errors = utils.squash_constants(c.stderr) if c.stderr else ""
        model_in_package_id = get_model_in_package_id(m)
        compilation_args = CompileTool().get_argument_list(parser, plugin_config_path, m.model_path)
        compilation_args = " ".join(map(str, compilation_args))

        row = [m.source, m.package_type, m.model_name, m.framework, m.precision,
               status, c.execution_time_ms, errors, c.stdout, report_id, model_in_package_id, compilation_args]
        compilation_table.append_row(row)

    return compilation_table


def extract_benchmark_info(stdout):
    benchmark_info = {}

    keywords = ["Median", "Average", "Min", "Max", "Throughput"]
    for keyword in keywords:
        match = re.search(f"{keyword}:\s*(\d*\.\d+|\d+)", stdout)
        if match:
            value = float(match.group(1))
            benchmark_info[keyword] = value
        else:
            benchmark_info[keyword] = None

    return benchmark_info


def inference(inference_list, models_list, report_id):
    inference_table = DataTable(
        "Inference", ["source", "package", "topology", "framework", "precision", "status", "executionTimeMs", "median", "average", "min",
                      "max", "throughputFps", "errorId", "logs", "reportId", "modelInPackageId"])

    models_meta = utils.get_models_meta(models_list)
    for i, m in zip(inference_list, models_meta):
        status = "PASSED" if i.return_code == 0 else "FAILED"
        model_in_package_id = get_model_in_package_id(m)
        info = extract_benchmark_info(i.stdout)

        row = [m.source, m.package_type, m.model_name, m.framework, m.precision,
               status, i.execution_time_ms, info["Median"], info["Average"], info["Min"], info["Max"], info["Throughput"],
               i.stderr, i.stdout, report_id, model_in_package_id]
        inference_table.append_row(row)

    return inference_table


def accuracy(accuracy_list, models_meta, report_id):
    accuracy_table = DataTable(
        "Accuracy", ["source", "package", "topology", "framework", "precision", "status", "metric", "value", "cpuValue", "groundTruth", "metricHint", "datasetSize", "errorId", "logs", "reportId", "modelInPackageId"])

    for a, m in zip(accuracy_list, models_meta):
        model_in_package_id = get_model_in_package_id(m)
        status = "PASSED" if a.return_code == 0 else "FAILED"
        row_head = [m.source, m.package_type,
                    m.model_name, m.framework, m.precision, status]
        row_tail = [a.stderr, a.stdout, report_id, model_in_package_id]

        if len(a.metrics) == 0:
            row_middle = [""] * 6
            row = row_head + row_middle + row_tail
            accuracy_table.append_row(row)
        else:
            for m in a.metrics:
                row_middle = [m.name, m.value, m.cpu_value,
                              m.ground_truth, m.metric_hint, m.dataset_size]
                row = row_head + row_middle + row_tail
                accuracy_table.append_row(row)

    return accuracy_table


def conformance(compile_list, models_list, parser, plugin_config_path):
    conformance_table = DataTable(
        "ConformanceCompilation", ["layer", "subgraph", "topology", "framework", "modelPrecision", "status", "errorId", "logs", "args"])

    models_meta = utils.get_models_meta(models_list)
    for c, m in zip(compile_list, models_meta):
        status = "PASSED" if c.return_code == 0 else "FAILED"
        errors = utils.squash_constants(c.stderr) if c.stderr else ""
        compilation_args = CompileTool().get_argument_list(parser, plugin_config_path, m.model_path)
        compilation_args = " ".join(map(str, compilation_args))

        row = [m.subgraph_layer, m.subgraph_name, m.model_name, m.framework,
               m.precision, status, errors, c.stdout, compilation_args]
        conformance_table.append_row(row)

    return conformance_table


def extract_imd_info(stdout):
    imd_info = {}

    patterns = [
        r"Median \((cc)\) : IRP: (\d+) IR: (\d+) IM: (\d+) over (\d+) runs\s+-\sVPU @ (\d+) MHz",
        r"Median \((FPS)\): IRP: ([\d.]+) IR: ([\d.]+) IM: ([\d.]+)",
        r"Median \((us)\) : IRP: ([\d.]+) IR: ([\d.]+) IM: ([\d.]+)"
    ]

    lines = stdout.strip().split("\n")
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                values = match.groups()
                imd_info[values[0]] = [float(values[1]), float(values[2]), float(values[3])]
                if len(values) > 4:
                    imd_info["num_runs"] = int(values[4])
                    imd_info["mhz"] = int(values[5])

    return imd_info


def inference_manager_demo(imd_list, models_list, report_id, parser):
    imd_table = DataTable(
        "IMD", ["source", "package", "topology", "framework", "precision", "status", "executionTimeMs",
                "num_runs", "mhz", "ccIRP", "ccIR", "ccIM", "fpsIRP", "fpsIR", "fpsIM", "usIRP", "usIR", "usIM",
                "errorId", "logs", "reportId", "modelInPackageId"])

    models_meta = utils.get_models_meta(models_list)
    for i, m in zip(imd_list, models_meta):
        errors = i.stderr if i.stderr else ""
        model_in_package_id = get_model_in_package_id(m)
        imd_info = extract_imd_info(i.stdout)

        num_runs = imd_info.get("num_runs", None)
        mhz = imd_info.get("mhz", None)
        cc = imd_info.get("cc", [None] * 3)
        fps = imd_info.get("FPS", [None] * 3)
        us = imd_info.get("us", [None] * 3)
        perf_list = [num_runs, mhz] + cc + fps + us

        passed = (i.return_code == 0) and (None not in perf_list)
        status = "PASSED" if passed else "FAILED"

        row = [m.source, m.package_type, m.model_name, m.framework, m.precision, status,
               i.execution_time_ms] + perf_list + [errors, i.stdout, report_id, model_in_package_id]
        imd_table.append_row(row)

    return imd_table
