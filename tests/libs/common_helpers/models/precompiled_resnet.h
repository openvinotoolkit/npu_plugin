//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include "model_loader.h"

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
    const std::string resnet_50_folder = ModelLoader_Helper::getTestModelsPath() + "/KMB_models/BLOBS/resnet-50/schema-3.24.3/";
    const std::string resnet_50_IR_folder = ModelLoader_Helper::getTestModelsPath() + "/KMB_models/INT8/public/ResNet-50/";

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
                resnet_50_IR_folder + "resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8.xml"
            };
};
