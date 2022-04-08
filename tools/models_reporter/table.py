#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import numpy as np
import pandas as pd
from itertools import groupby
from tqdm.contrib.concurrent import process_map
import itertools

import utils


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
                "Table row width doesn't match to number of columns")

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
        return df.sort_values(by=column_headers)


def report(device, setup_config, plugin_config_list, report_id):
    report_table = DataTable(
        "Report", ["device", "ovBranch", "ovRevision", "vpuxBranch", "vpuxRevision", "vpuxCommitDate", "pluginConfig", "reportId"])

    ov_branch = setup_config.ov_branch
    ov_revision = setup_config.ov_commit
    vpux_branch = setup_config.vpux_branch
    vpux_revision = setup_config.vpux_commit
    vpux_commit_date = setup_config.vpux_commit_date
    plugin_config = ",".join(plugin_config_list)

    row = [device, ov_branch, ov_revision,
           vpux_branch, vpux_revision, vpux_commit_date, plugin_config, report_id]
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


def layers_in_model_func(args):
    model_layers, model_in_package_id = args
    layers_with_params = []
    for layer in model_layers:
        layer_id = create_primary_key([layer.name, layer.opset])

        row = [layer.input_shapes, layer.output_shapes,
               layer.attributes, model_in_package_id, layer_id]
        layers_with_params.append(row)

    return layers_with_params


def layer_in_model(model_layers_list, models_meta):
    layer_in_model_table = DataTable(
        "LayerInModel", ["inputShapes", "outputShapes", "attributes", "modelInPackageId", "layerId"])

    ids = [get_model_in_package_id(model_meta) for model_meta in models_meta]

    input = [(layers, id) for layers, id in zip(model_layers_list, ids)]

    print("Glob all layer parameters...")
    list_of_model_layers = process_map(
        layers_in_model_func, input, max_workers=os.cpu_count(), chunksize=1)

    print("Removing duplicates...")
    for model_layers in list_of_model_layers:
        for layer in model_layers:
            layer_in_model_table.append_row(layer)

    return layer_in_model_table.remove_duplicates()


def unsupported_layer(models_meta, unsupported_models_layers):
    unsupported_layer_table = DataTable(
        "UnsupportedLayer", ["source", "package", "topology", "framework", "precision", "name", "opset", "reason", "layerId", "modelInPackageId"])

    for m, unsupported_model_layers in zip(models_meta, unsupported_models_layers):
        model_in_package_id = get_model_in_package_id(m)
        for name, opset, reason in unsupported_model_layers:
            layer_id = create_primary_key([name, opset])
            new_row = [m.source, m.package_type, m.model_name, m.framework,
                       m.precision, name, opset, reason, layer_id, model_in_package_id]
            unsupported_layer_table.append_row(new_row)

    return unsupported_layer_table


def compilation(compile_list, report_id):
    compilation_table = DataTable(
        "Compilation", ["source", "package", "topology", "framework", "precision", "status", "compileTimeMs", "errorId", "logs", "reportId", "modelInPackageId"])

    for c in compile_list:
        status = "PASSED" if c.compiled else "FAILED"
        compile_time_ms = c.compile_time_ms
        errors = utils.squash_constants(c.stderr) if c.stderr else ""
        logs = c.stdout if c.stdout else ""

        m = c.model_meta
        model_in_package_id = get_model_in_package_id(m)

        row = [m.source,  m.package_type,  m.model_name,  m.framework,  m.precision,
               status, compile_time_ms, errors, logs, report_id, model_in_package_id]
        compilation_table.append_row(row)

    return compilation_table


def inference(inference_list, report_id):
    inference_table = DataTable(
        "Inference", ["source", "package", "topology", "framework", "precision", "status", "inferenceTimeMs", "errorId", "logs", "reportId", "modelInPackageId"])

    for i in inference_list:
        inference_time_ms = i.execution_time_ms
        status = "PASSED" if i.executed else "FAILED"
        errors = utils.squash_constants(i.stderr) if i.stderr else ""
        logs = i.stdout if i.stdout else ""

        m = i.model_meta
        model_in_package_id = get_model_in_package_id(m)

        row = [m.source,  m.package_type,  m.model_name,  m.framework,  m.precision,
               status, inference_time_ms, errors, logs, report_id, model_in_package_id]

        inference_table.append_row(row)

    return inference_table
