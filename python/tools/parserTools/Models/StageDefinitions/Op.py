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

from Controllers.EnumController import *

class Op:

    def __init__(self, name):
        self.name = name

        self.requirements = {
            "input": {
                "layout": StorageOrder.orderYXZ
            },
            "weights":{
                "layout": TapsOrder.orderHWCK
            },
            "Hardware": False
        }

    # The following functions are declared to error in absence of being overwritten
    def pushStageDetailsToObject(self):
        raise NotImplementedError("The method not implemented")

    def changeLayoutRequirements(self, key, new_layout):
        assert key in self.requirements
        self.requirements[key]["layout"] = new_layout
