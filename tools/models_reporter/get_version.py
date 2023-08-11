#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import re
from tools import parallel
from tools.compile_tool import CompileTool
from pathlib import Path


def parse_openvino_version(stdout):
    openvino_version_regex = re.compile(r"^Build \.+ (.*?)$")
    for line in stdout.splitlines():
        match = openvino_version_regex.match(line)
        if match:
            return match.groups()[0]
    return "Not found"


def parse_vpux_plugin_version(stdout):
    vpux_version_regex = re.compile(r".*?\[VPUIP::BackEnd\]Blob version:.*?hash=(.*?),.*$")
    for line in stdout.splitlines():
        match = vpux_version_regex.match(line)
        if match:
            return match.groups()[0]
    return "Not found"


def get_version_by_compilation(binaries, output_dir):
    empty_model = Path("empty_net.xml")
    env = CompileTool().get_environment(binaries)
    env["IE_VPUX_LOG_LEVEL"] = "LOG_INFO"
    args = [binaries / "compile_tool",
            "-m", empty_model,
            "-d", "VPUX.3720",
            "-o", output_dir / "empty_net.blob"]

    result = parallel.one_model(args, env, timeout=10)
    ov_version = parse_openvino_version(result.stdout)
    vpux_version = parse_vpux_plugin_version(result.stdout)
    return ov_version, vpux_version
