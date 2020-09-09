//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include "test_model_path.hpp"

struct modelBlobInfo {
    std::string graphName, graphPath, inputPath, outputPath;
    std::string nv12Input, nv12Output;
    std::string nv12_1080Input, nv12_1080Output;
    std::string modelPath;
};

struct modelTensors {
    InferenceEngine::TensorDesc inputTensor, outputTensor;
};

namespace PrecompiledResNet_Helper {
    const std::string resnet_50_folder = ModelsPath() + "/KMB_models/BLOBS/resnet-50/";
    const std::string resnet_50_IR_folder = ModelsPath() + "/KMB_models/INT8/public/ResNet-50/";

    // Actual version, fp16 output
    static const modelBlobInfo resnet50 = {
                "resnet-50",
                resnet_50_folder + "resnet-50.blob",
                resnet_50_folder + "input.bin",
                resnet_50_folder + "output.bin",
                resnet_50_folder + "input-228x228-nv12.bin",
                resnet_50_folder + "output-228x228-nv12.bin",
                "input-cat-1080x1080-nv12.bin",
                "output-cat-1080x1080-nv12.bin",
                resnet_50_IR_folder + "resnet50_uint8_int8_weights_pertensor.xml"
            };
};

