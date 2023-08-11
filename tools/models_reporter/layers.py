#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
import sys

from tqdm.contrib.concurrent import process_map
import xml.etree.ElementTree as xml_parser

import utils


@dataclass
class Layer(utils.StrictTypeCheck):
    name: str
    opset: int
    attributes: str
    input_shapes: list
    output_shapes: list

    def __repr__(self):
        inputs = self.input_shapes
        outputs = self.output_shapes
        attrs = self.attributes
        return f"{self.name}-{self.opset} inputs:{inputs} outputs:{outputs} {attrs}"


def parse_shapes(layer, tag):
    shapes = []
    edges = layer.findall(tag)
    if edges:
        for edge in edges:
            ports = edge.findall('port')
            for port in ports:
                dims = port.findall('dim')
                shape = [int(dim.text) for dim in dims] if dims else [1]
                shapes.append(shape)
    return shapes


def parse_model_proc_func(model_path):
    ir_tree = xml_parser.parse(model_path)
    ir_root = ir_tree.getroot()

    layers_info = []

    for layer in ir_root.find('layers'):
        name = layer.attrib.get('type')
        opset = layer.attrib.get('version')[len('opset'):]
        data = layer.find('data')

        input_shapes = parse_shapes(layer, 'input')
        output_shapes = parse_shapes(layer, 'output')
        attributes = dict(sorted(data.attrib.items())) if data is not None else dict()

        layer = Layer(name, int(opset), str(attributes), input_shapes, output_shapes)
        layers_info.append(layer)

    return layers_info


def glob_all_layers(models_list, max_workers):
    model_layers_list = process_map(
        parse_model_proc_func, models_list, max_workers=max_workers, chunksize=4)

    return model_layers_list


if __name__ == "__main__":
    parse_model_proc_func(sys.argv[1])
