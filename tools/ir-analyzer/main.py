#!/usr/bin/env python3
import argparse
import xmltodict
import typing
import csv
import logging

layers_to_exclude = ["Const"]


def parse_args():
    parser = argparse.ArgumentParser(description="Tool for extracting layers info from IR")
    parser.add_argument('path_to_model', type=str,
                        help='Required. Path to an .xml file with a trained model.')
    parser.add_argument('-o', '--path_to_output', type=str, required=False, default="output_data.csv",
                        help='Optional. Path to output file with statistics. (default: output_data.csv)')
    parser.add_argument('-rd', '--remove_duplicates', dest='remove_duplicates', action='store_true',
                        help='Optional. Eliminate duplicated combinations from output. (default: enabled)')
    parser.add_argument('-nrd', '--not_remove_duplicates', dest='remove_duplicates', action='store_false',
                        help='Optional. Leave duplicated combinations. (default: disabled)')
    parser.set_defaults(remove_duplicates=True)
    parser.add_argument('-s', '--sort', dest='sort', action='store_true',
                        help='Optional. Sort resulting combinations. (default: enabled)')
    parser.add_argument('-ns', '--not_sort', dest='sort', action='store_false',
                        help='Optional. Not sort resulting combinations. (default: disabled)')
    parser.set_defaults(sort=True)
    return parser.parse_args()


def get_layout_from_ndims(ndims: int) -> typing.List[str]:
    if ndims == 0:
        return []
    elif ndims == 1:
        return ["C"]
    elif ndims == 2:
        return ["N", "C"]
    elif ndims == 3:
        return ["C", "H", "W"]
    elif ndims == 4:
        return ["N", "C", "H", "W"]
    elif ndims == 5:
        return ["N", "C", "D", "H", "W"]
    else:
        logging.warning("Unsupported number of dims! ndims > 5 is unsupported. ndims = {}".format(ndims))
        return ["UNDEFINED"]


def get_inputs(root: dict) -> typing.List[dict]:
    return get_ports(root.get("input", {}))


def get_outputs(root: dict) -> typing.List[dict]:
    return get_ports(root.get("output", {}))


def get_attributes(root: dict) -> dict:
    return root.get("data", [])


def get_ports(root: dict) -> typing.List[dict]:
    ports = root.get("port", [])
    if not isinstance(ports, list):
        ports = [ports]
    return ports


def get_dims(root: dict) -> typing.List[str]:
    dims = root.get("dim", [])
    if isinstance(dims, str):
        dims = [dims]
    return dims


def get_layers(root: dict) -> typing.List[dict]:
    return root.get("layers", {}).get("layer", [{}])


def get_type(root: dict) -> str:
    return root.get("@type")


def get_opset(root: dict) -> str:
    return root.get("@version")


def get_precision(root: dict) -> str:
    return root.get("@precision", "")


def get_layer_info(root: dict) -> typing.List[typing.List[str]]:
    if get_type(root) not in layers_to_exclude:
        output_data = []
        layer_info = []
        layer_info.append(get_type(root))
        layer_info.append(get_opset(root))
        inputs = get_inputs(root)
        outputs = get_outputs(root)
        attributes = get_attributes(root)
        if inputs:
            for input_info in inputs:
                dims = get_dims(input_info)
                layout = get_layout_from_ndims(len(dims))
                layer_info.append('in: ' + str(list(map(lambda x: int(x), dims))) + ' (' + ''.join(layout) + " " + get_precision(input_info) + ')')
        if outputs:
            for output_info in outputs:
                dims = get_dims(output_info)
                layout = get_layout_from_ndims(len(dims))
                layer_info.append('out: ' + str(list(map(lambda x: int(x), dims))) + ' (' + ''.join(layout) + " " + get_precision(output_info) + ')')
        if attributes:
            for k, v in attributes.items():
                layer_info.append('='.join([k[1:],v]))
        output_data.append(layer_info)
        if "body" in root.keys():
            layers = get_layers(root["body"])
            for layer in layers:
                sublayers_info = get_layer_info(layer)
                if sublayers_info:
                    output_data.extend(sublayers_info)
        return output_data


def get_layers_info(root: dict, is_only_unique: bool, need_sort: bool) -> typing.List[typing.List[str]]:
    layers = get_layers(root["net"])
    output_data = []
    for layer in layers:
        layer_info = get_layer_info(layer)
        if layer_info:
            output_data.extend(layer_info)
    if is_only_unique:
        output_data = list(set(map(lambda x: tuple(x), output_data)))
    if need_sort:
        output_data.sort()
    return output_data


if __name__ == "__main__":
    args = parse_args()

    with open(args.path_to_model) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    output_data = get_layers_info(data_dict, args.remove_duplicates, args.sort)

    with open(args.path_to_output, "w") as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerows(output_data)
