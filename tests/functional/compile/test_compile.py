#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for IR compilation.
Refer to conftest.py on the test usage.
"""

import os
import subprocess


def test_compile(param_ir):
    """ Test network can be compiled
    """
    out = os.path.splitext(os.path.join(param_ir.output_dir, param_ir.model))[0] + ".blob"
    os.makedirs(os.path.split(out)[0])
    subprocess.check_call(
        [param_ir.compiler_tool,
         '-d=KMB',
         f'-m={os.path.join(param_ir.models_dir, param_ir.model)}',
         f'-o={out}'])
