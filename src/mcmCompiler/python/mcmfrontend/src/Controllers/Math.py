#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from functools import reduce


def prod(x):
    # Returns the result of multiplying each element of a list
    if x:
        return reduce(lambda a, b: a * b, x, 1)
    else:
        return 1
