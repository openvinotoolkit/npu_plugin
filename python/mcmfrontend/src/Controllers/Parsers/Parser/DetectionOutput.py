
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

import numpy as np

from .Layer import Layer
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat


class DetectionOutput(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        tfPL = TensorFormat(Layouts.NCHW, (1,))
        self.formatPool = [(tfCM, tfPL)]

    def loadTensorSizes(self, tensorDict):
        super().loadTensorSizes(tensorDict)

        # Because Myriad cannot return a variable tensor size,
        # we force the output of this layer to be the maximum possible
        self.outputTensorSizes = []
        assert(len(self.outputTensorNames) == 1)
        for name in self.outputTensorNames:
            self.outputTensorSizes.append((1, 1, self.getMaxDetections(), 7))

    def load_parameters(self, detection_param):
        share_location = 1 if detection_param.share_location else 0

        det_out_dtype = np.dtype(
            "<i4, <i4, <i4, <f4, <i4, <i4, <i4, <f4, <i4, <f4")

        self.params = np.array((detection_param.num_classes,
                                share_location,
                                detection_param.background_label_id,
                                detection_param.nms_param.nms_threshold,
                                detection_param.nms_param.top_k,
                                detection_param.code_type,
                                detection_param.keep_top_k,
                                detection_param.confidence_threshold,
                                detection_param.variance_encoded_in_target,
                                detection_param.nms_param.eta), det_out_dtype)

        self.params = self.params.flatten()
        self.params = self.params.view("<f4")

        self.max_detections = detection_param.keep_top_k + 1

    def getMaxDetections(self):
        return self.max_detections
