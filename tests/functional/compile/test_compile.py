#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for IR compilation.
Refer to conftest.py on the test usage.
"""

import os
import subprocess
import tempfile
import logging


def run(args, log=None, verbose=True):
    """ Run command
    """
    if log is None:
        log = logging.getLogger()
    log_out = log.info if verbose else log.debug

    log_out(f'========== cmd: {" ".join(args)}')  # pylint: disable=logging-format-interpolation,logging-fstring-interpolation

    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            encoding='utf-8',
                            universal_newlines=True)
    output = []
    for line in iter(proc.stdout.readline, ''):
        log_out(line.strip('\n'))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log_out(outs.strip('\n'))
        output.append(outs)
    log_out('========== Completed. Exit code: %d', proc.returncode)
    return proc.returncode, ''.join(output)


def test_compile(request, param_ir):
    """ Test network can be compiled
    """
    out = os.path.splitext(os.path.join(param_ir.output_dir, param_ir.model))[0] + ".blob"
    os.makedirs(os.path.split(out)[0])
    with tempfile.NamedTemporaryFile() as time_file:
        returncode, output = run(
            ['/usr/bin/time', '--format=%M', f'--output={time_file.name}', '--quiet',
             param_ir.compiler_tool,
             '-d=KMB',
             f'-m={os.path.join(param_ir.models_dir, param_ir.model)}',
             f'-o={out}'])
        print(output)
        peak_memory = open(time_file.name).read().strip()
        request.node.peak_memory = peak_memory
        print(f'Peak memory consumption {peak_memory}Kb')
        assert returncode == 0, f'Command exited with non-zero status {returncode}'
