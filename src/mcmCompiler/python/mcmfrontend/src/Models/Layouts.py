# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

"""
    For values that NEVER change.
    Also used for quicker aliases of more complicated structures.
"""


"""
    ### Layout Constants ###


    ## NCHW Notation
    Notation as per Caffe and other Neural Net Frameworks.
    Axes labels are as such:
    N - Batch Number
    C - Channels
    H - Height
    W - Width

    This notation does not transfer to all Tensors (e.g. Weights), but
    should still be used as reference.

    ## "Major, Minor" Notation:

    The axis which changes fastest / has the least distance between elements
    in a matrix is the 'minor' axis.

    The axis which changes slowest / has the largest distance between elements
    in a matrix is the 'major' axis.

    if the remaining axes are following canonical layout (i.e. incrementing)
    we take this to be the 'standard' version.
    if they are flipped, we consider this an 'alternate' representation.
"""

"""
    The format commonly known as Channel Minor:
    It is a common software layout
    Note: There are technically 2 layouts that could be called
    'Channel Minor'. This is the particular one we use.
"""
NHWC = (0, 2, 3, 1)

"""
    Planar is the main way of thinking about data.
    It is also the format referred to as Canonical in Tensor Terminology
    This is the default layout of Caffe and one of the operable formats for MyriadX
"""
NCHW = (0, 1, 2, 3)

"""
    Row interleaved is a format used in MyriadX Hardware for alternative
    access patterns.
"""
NHCW = (0, 2, 1, 3)

# The following are not used, but are present to allow for easier development
# if they do enter common usage:

"""
    Width and Height are swapped.
"""
NCWH = (0, 1, 3, 2)

"""
    Axes are the reverse of the Planar format, but batchsize is preserved.
"""
NWHC = (0, 3, 2, 1)

"""
    Name based on standards at the top of this files.
"""
NWCH = (0, 3, 1, 2)

"""
    Kmb layouts
"""
ChannelMajor = (0, 1, 2, 3)
ZMajor = (0, 2, 3, 1)

"""
    Kmb weights are in the format KWHC
    (canonical weights layout are in KCHW)
"""
SparsityTensorLayout = (0, 1, 2, 3)
WeightsTableLayout = (0, 3, 2, 1)
