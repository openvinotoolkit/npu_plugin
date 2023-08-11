#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import os
import re
import errno
from typing import Union
from enum import Enum

from PIL import Image
import random
import platform
import xml.etree.ElementTree as ET
import shutil


@dataclass
class StrictTypeCheck:
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(
                    f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")


@dataclass
class ModelMeta(StrictTypeCheck):
    model_path: Path
    model_path_relative: Path
    ir_name: str
    precision: str
    framework: str
    model_name: str
    package_name: str
    package_type: str
    source: str
    # Subgraph information
    subgraph_data_type: str
    subgraph_name: str
    subgraph_layer: str
    batch: int = 1  # openvino models


def squash_constants(stderr_str):
    pattern = re.compile(r'{[\d\.:,]+}')
    return re.sub(pattern, r"{_CONST_}", stderr_str)


def match_package_name_vpu(package_name):
    pattern = re.compile(r'^\d{8}_vpu-models-(\w+)-ir_v\d+.*$')
    return pattern.match(package_name)


def match_package_name_openvino(package_name):
    pattern = re.compile(r'^ww\d+_(\w+)_.*$')
    return pattern.match(package_name)


def get_package_type(package_name):
    match = match_package_name_vpu(package_name)
    if match is None:
        match = match_package_name_openvino(package_name)
        if match is None:
            return "UNKNOWN"

    package_type = str(match.groups()[0]).upper()
    if package_type in ['KPI', 'POR']:
        return 'POR'
    elif package_type in ['FULL', 'SCALE']:
        return 'SCALE'
    return "UNKNOWN"


def get_model_topology_from_subgraph(model_path):
    with open(model_path, 'r') as f:
        for line in f:
            if ".xml" in line:
                match = re.search(r'after_ngraph\\([^\\]*)', line)
                if match:
                    subgraph_topology = match.group(1)

                    return subgraph_topology
                else:
                    return "No match found for subgraph topology"
    return "Missing meta file"


def get_model_framework_from_subgraph(model_path, subgraph_topology):
    with open(model_path, 'r') as f:
        for line in f:
            if ".xml" in line:
                match = re.search(rf'%s\\([^\\]*)' % subgraph_topology, line)
                if match:
                    subgraph_framework = match.group(1)

                    return subgraph_framework
                else:
                    return "No match found for subgraph framework"
    return "Missing meta file"


def get_model_precision_from_subgraph(model_path):
    with open(model_path, 'r') as f:
        for line in f:
            if ".xml" in line:
                match = re.search(r'\bFP16(-INT8)?\b', line)
                if match:
                    subgraph_precision = match.group(0)

                    return subgraph_precision
                else:
                    return "No match found for subgraph precision"
    return "Missing meta file"


def date_to_work_week(year, month, day):
    (_, work_week, work_day) = datetime(year, month, day).isocalendar()
    return work_week, work_day


def get_work_week(package_name):
    pattern_vpu = re.compile(r'^(\d{8})_vpu-models-.*$')
    pattern_ov = re.compile(r'^ww(\d+)_\w+_.*$')

    match = pattern_vpu.match(package_name)
    if match:
        date_string = match.groups()[0]
        year = int(date_string[:4])
        month = int(date_string[4:6])
        day = int(date_string[6:])
        work_week, work_day = date_to_work_week(year, month, day)
        return work_week

    match = pattern_ov.match(package_name)
    if match:
        work_week = match.groups()[0]
        return int(work_week)

    return "UNKNOWN"


def get_source(path):
    if '_vpu-models-' in str(path):
        return 'VPU'
    elif 'conformance' in str(path):
        return 'conformance'
    else:
        return 'OpenVINO'


class OVModel(Enum):
    FP32 = 1
    FP16 = 2
    DLDT_INT8 = 3
    OPTIMIZED_INT8 = 4


def get_openvino_model_type(path):
    if path.parents[2].name == 'FP32':
        return OVModel.FP32
    elif path.parents[0].name == 'dldt':
        return OVModel.FP16
    elif path.parents[0].name == 'dldt_int8':
        return OVModel.DLDT_INT8

    return OVModel.OPTIMIZED_INT8


@lru_cache(maxsize=None)
def metadata_from_path(model_path):
    model_path = Path(model_path).absolute()
    source = get_source(model_path)

    if source == 'VPU':
        normal_path_match = match_package_name_vpu(model_path.parents[3].name)
        package_root_path = model_path.parents[3] if normal_path_match else model_path.parents[4]
        package_name = package_root_path.name
        model_path_relative = model_path.relative_to(package_root_path.parent)

        return ModelMeta(
            model_path=model_path,
            model_path_relative=model_path_relative,
            ir_name=model_path.name,
            precision=model_path.parents[0].name,
            framework=model_path.parents[1].name,
            model_name=model_path.parents[2].name,
            package_name=package_name,
            package_type=get_package_type(package_name),
            source=get_source(package_name),
            subgraph_data_type="",
            subgraph_name="",
            subgraph_layer="")
    elif source == 'conformance':
        package_root_path = model_path.parents[2]
        package_name = package_root_path.name
        model_path_relative = model_path.relative_to(package_root_path.parent)
        meta_path = model_path.with_suffix(".meta")

        return ModelMeta(
            model_path=model_path,
            model_path_relative=model_path_relative,
            ir_name=model_path.name,
            precision=get_model_precision_from_subgraph(meta_path),
            framework=get_model_framework_from_subgraph(
                meta_path, get_model_topology_from_subgraph(meta_path)),
            model_name=get_model_topology_from_subgraph(meta_path),
            package_name=package_name,
            package_type=get_package_type(package_name),
            source=get_source(package_name),
            subgraph_data_type=model_path.parents[0].name,
            subgraph_layer=model_path.parents[1].name,
            subgraph_name=model_path.name.replace(".xml", ""))
    else:
        ov_model_type = get_openvino_model_type(model_path)
        package_root_path = model_path.parents[8] if ov_model_type == OVModel.OPTIMIZED_INT8 else model_path.parents[6]
        package_name = package_root_path.name
        model_path_relative = model_path.relative_to(package_root_path.parent)

        if ov_model_type == OVModel.OPTIMIZED_INT8:
            return ModelMeta(
                model_path=model_path,
                model_path_relative=model_path_relative,
                ir_name=model_path.name,
                precision=model_path.parents[4].name +
                '-' + model_path.parents[3].name,
                framework=model_path.parents[6].name,
                model_name=model_path.parents[7].name,
                package_name=package_name,
                package_type=get_package_type(package_name),
                source=get_source(package_name),
                batch=int(model_path.parents[2].name),
                subgraph_data_type="",
                subgraph_name="",
                subgraph_layer="")
        else:
            return ModelMeta(
                model_path=model_path,
                model_path_relative=model_path_relative,
                ir_name=model_path.name,
                precision=model_path.parents[2].name,
                framework=model_path.parents[4].name,
                model_name=model_path.parents[5].name,
                package_name=package_name,
                package_type=get_package_type(package_name),
                source=get_source(package_name),
                batch=int(model_path.parents[1].name),
                subgraph_data_type="",
                subgraph_name="",
                subgraph_layer="")


def get_models_meta(models_list):
    return [metadata_from_path(path) for path in models_list]


def relative_change_dir(output_dir, relative_path):
    output_path = os.path.dirname(os.path.join(output_dir, relative_path))
    os.makedirs(output_path, exist_ok=True)
    os.chdir(output_path)


def flatten(lists):
    return [item for list in lists for item in list]


def get_path(entry: Union[str, Path], is_directory=False, check_exists=True, file_or_directory=False):
    try:
        path = Path(entry)
    except TypeError as type_err:
        raise TypeError(
            '"{}" is expected to be a path-like'.format(entry)) from type_err
    if not check_exists:
        return path
    # Path.exists throws an exception in case of broken symlink
    if not os.path.exists(str(path)):
        raise FileNotFoundError('{}: {}'.format(
            os.strerror(errno.ENOENT), path))
    if not file_or_directory:
        if is_directory and not path.is_dir():
            raise NotADirectoryError('{}: {}'.format(
                os.strerror(errno.ENOTDIR), path))
        # if it exists it is either file (or valid symlink to file) or directory (or valid symlink to directory)
        if not is_directory and not path.is_file():
            raise IsADirectoryError('{}: {}'.format(
                os.strerror(errno.EISDIR), path))
    return path


def model_to_blob_path(model_path: Path, output_dir: Path):
    meta = metadata_from_path(model_path)
    relative_path = meta.model_path_relative.with_suffix(".blob")
    return output_dir / relative_path


def glob_files(models_path: Path, wildcard):
    return [path for path in models_path.rglob(wildcard)]


def safe_to_save_filename(path: str):
    return re.sub(r'[\<\>\:\"\|\?\*]', '_', path)


def generate_random_gradient_image(C, H, W, model):
    path = os.path.dirname(model)
    if os.path.exists(f'{path}/random_gradient_image.png'):
        print(f'Image already exists for {os.path.basename(model)}')
    else:
        print("Generate random gradient input...")
        img = Image.new("RGBA", (W, H), 0)
        pixels = img.load()

        for x in range(W):
            for y in range(H):
                for c in range(C):
                    r = int(255 * (x / W))
                    g = int(255 * (y / H))
                    b = int(255 * (x + y) / (W + H))
                    pixels[x, y] = (
                        r,
                        g,
                        b,
                        255,
                    )
                    pixels[x, y] = tuple(
                        int(C * random.random()) for C in pixels[x, y]
                    )
        path = os.path.dirname(model)
        img.save(f'{path}/random_gradient_image.png')


def get_input_shape_model(model):
    N_in, C_in, H_in, W_in = 1, 1, 1, 1
    with open(model, 'r') as f:
        for line in f:
            if 'input_shape value=' in line:
                pattern = r'\[([^]]+)\]'
                match = re.search(pattern, line)
                if match:
                    values_str = match.group(1)
                    values = [int(x.strip()) for x in values_str.split(',')]
                    if len(values) == 2:
                        H_in, W_in = values
                    elif len(values) == 3:
                        C_in, H_in, W_in = values
                    elif len(values) == 4:
                        N_in, C_in, H_in, W_in = values

                    if C_in > W_in:
                        C_in, W_in = W_in, C_in

    return N_in, C_in, H_in, W_in


def get_output_shape_model(model):

    tree = ET.parse(model)
    root = tree.getroot()

    dims = []

    for layer in root.iter('layer'):
        if layer.get('type') == 'Result':
            for port in layer.iter('port'):
                values = []
                for dim in port.iter('dim'):
                    values.append(int(dim.text))
            dims.append(values)

    return dims


def find_matching_xml(models_list, blob_path):
    model_name = os.path.normpath(blob_path).split(os.sep)
    if model_name[-1] != 'dldt':
        model_slice = model_name[-7:-3]
    else:
        model_slice = model_name[-8:-4]
    match = os.sep.join(model_slice)
    network = match.replace(os.sep,'-')

    for xml_file in models_list:
        if match in str(xml_file):
            return xml_file, str(network)

    return None


def verify_matching_blobs(blob_cpu, blob_vpu):
    blob_cpu = Path(blob_cpu).parent
    blob_vpu = Path(blob_vpu).parent
    return blob_cpu == blob_vpu


def get_blob_file_paths(directory_path):
    cpu_filename = []
    vpu_filename = []

    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.blob'):
                if '_CPU' in filename and 'ref_out_0' in filename:
                    cpu_filename.append(os.path.join(dirpath, filename))
                elif '_VPU' in filename and 'ref_out_0' in filename:
                    vpu_filename.append(os.path.join(dirpath, filename))
    return cpu_filename, vpu_filename


def append_device_to_blob_name(blobs, device):
    for root, dirs, files in os.walk(blobs):
        for file in files:
            # Check if the file has a .blob extension
            if file.endswith('ref_out_0_case_0.blob') and device not in file:
                # Construct the new file name by appending "_device" to the original name
                new_file_name = os.path.splitext(
                    file)[0] + device + os.path.splitext(file)[1]

                # Construct the full paths to the old and new files
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name)

                # Rename the file
                shutil.move(old_file_path, new_file_path)
