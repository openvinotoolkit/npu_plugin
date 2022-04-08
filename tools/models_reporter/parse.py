#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from itertools import zip_longest
import re
import utils
from dataclasses import astuple, dataclass
import query


def parse_missing_kernels(stdout):
    pattern = re.compile(r'.*?Missing SW.Kernel.*?\'(.*?)\'.*?at \'loc.*$')

    missing_layers = set()
    for line in stdout.splitlines():
        match = pattern.match(line)
        if match:
            missing_kernel_list = set(match.groups())
            missing_layers.update(missing_kernel_list)

    return list(missing_layers)


def find_name_and_opset(layer_names, all_layers_with_opsets):
    found = set()
    for layer_name in layer_names:
        try:
            layers = ((name, opset)
                      for name, opset in all_layers_with_opsets if name == layer_name)
            found.update(layers)
        except ValueError:
            found.add((layer_name, "unknown"))

    return list(found)


def collect_missing_vpu_kernels(missing_kernels_list, all_layers_with_opsets):
    missing_kernels = set(kernel[4:] for kernel in missing_kernels_list)
    return find_name_and_opset(missing_kernels, all_layers_with_opsets)


@dataclass
class UnsupportedLayer:
    name: str
    opset: str
    reason: str

    # to be able to unpack
    def __iter__(self):
        return iter(astuple(self))


def unsupported_with_error(stderr, all_layers_set):
    failed_layers = set()
    words = utils.flatten([line.split(" ") for line in stderr.splitlines()])
    for layer, opset in all_layers_set:
        if layer in words:
            failed_layers.add((layer, opset, stderr))

    return [UnsupportedLayer(layer, opset, reason) for (layer, opset, reason) in failed_layers]


def unsupported_with_reason(unsupported_ops, reason):
    return [UnsupportedLayer(name, opset, reason) for name, opset in unsupported_ops]


def unsupported_layer_with_error(ops_and_errors):
    return [UnsupportedLayer(name, opset, error) for (name, opset, error) in ops_and_errors]


def get_all_layers_set(model_layers_list):
    all_layers_set = set()
    for model_layers in model_layers_list:
        model_layers_set = ((layer.name, layer.opset)
                            for layer in model_layers)
        all_layers_set = all_layers_set.union(model_layers_set)
    return all_layers_set


def sw_kernels_missing(compiler_stdout, all_layers_with_opsets):
    missing_kernels_list = parse_missing_kernels(compiler_stdout)
    missing_kernels = collect_missing_vpu_kernels(
        missing_kernels_list, all_layers_with_opsets)
    return unsupported_with_reason(missing_kernels,
                                   "SW.Kernel is not implemented")


def query_model_result(query_stdout, all_layers_with_opsets):
    hint = "Unsupported Layers: "
    unsupported_layers_list = [line[len(hint):].split(',')
                               for line in query_stdout.splitlines() if hint in line]
    unsupported_layers_list = utils.flatten(unsupported_layers_list)
    unsupported = find_name_and_opset(
        unsupported_layers_list, all_layers_with_opsets)
    return unsupported_with_reason(unsupported, "Unsupported by query_model")


def query_model_legacy(query_stderr, all_layers_with_opsets):
    pattern = re.compile(r'^Cannot create (.*?) layer.*')
    match = pattern.match(query_stderr)
    unsupported_layers_set = (match.groups()[0]) if match else set()
    legacy_ops = find_name_and_opset(
        unsupported_layers_set, all_layers_with_opsets)
    return unsupported_with_reason(legacy_ops, "Legacy operation")


def layers_failed_to_compile(compiler_stderr, all_layers_with_opsets):
    compiler_stderr = utils.squash_constants(compiler_stderr)
    return unsupported_with_error(compiler_stderr, all_layers_with_opsets)


def unsupported_from_compile_info(compile_list, model_layers_list):
    models_unsupported_layers_list = []
    for compile_info, model_layers in zip(compile_list, model_layers_list):
        all_layers_with_opsets = get_all_layers_set([model_layers])
        unsupported = sw_kernels_missing(
            compile_info.stdout, all_layers_with_opsets)
        unsupported.extend(layers_failed_to_compile(
            compile_info.stderr, all_layers_with_opsets))

        models_unsupported_layers_list.append(unsupported)

    return models_unsupported_layers_list


def unsupported_from_query_model(query_list, model_layers_list):
    if len(query_list) == 0:
        return []

    models_unsupported_layers_list = []
    for query_info, model_layers in zip(query_list, model_layers_list):
        all_layers_with_opsets = get_all_layers_set([model_layers])
        unsupported = query_model_result(
            query_info.stdout, all_layers_with_opsets)
        unsupported.extend(query_model_legacy(
            query_info.stderr, all_layers_with_opsets))

        models_unsupported_layers_list.append(unsupported)

    return models_unsupported_layers_list


def join_lists_of_lists(list0, list1):
    return [compilation + query_model for compilation, query_model in zip_longest(list0, list1)]
