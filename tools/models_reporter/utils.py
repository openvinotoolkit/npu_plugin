#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from datetime import datetime
import os
import pathlib
import re


@dataclass
class ModelMeta:
    model_path: pathlib.Path
    model_path_relative: pathlib.Path
    ir_name: str
    precision: str
    framework: str
    model_name: str
    package_name: str
    package_type: str
    source: str
    batch: str = '1'  # openvino models


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
    return 'VPU' if '_vpu-models-' in str(path) else 'OpenVINO'


def metadata_from_path(model_path):
    model_path = pathlib.Path(model_path).absolute()
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
            source=get_source(package_name))
    else:
        is_optimized_model = (model_path.parents[0].name != 'dldt')
        package_root_path = model_path.parents[8] if is_optimized_model else model_path.parents[6]
        package_name = package_root_path.name
        model_path_relative = model_path.relative_to(package_root_path.parent)

        if is_optimized_model:
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
                batch=model_path.parents[2].name)
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
                batch=model_path.parents[1].name)


def relative_change_dir(output_root, relative_path):
    output_path = os.path.dirname(os.path.join(output_root, relative_path))
    os.makedirs(output_path, exist_ok=True)
    os.chdir(output_path)


def flatten(lists):
    return [item for list in lists for item in list]
