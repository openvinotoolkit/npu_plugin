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

# List of globally accessible objects

USING_MA2480 = False
#FIRST_STAGE_NEEDS_PADDING = False
INPUT_IN_INTERLEAVED = False
OPT_SCHEDULER = True
n_DPE = 256
CMX_SIZE = 512 * 1024


BLOB_MAJOR_VERSION = 2  # Architectural Changes
BLOB_MINOR_VERSION = 3  # Incompatible Changes
BLOB_PATCH_VERSION = 0  # Compatible Changes
