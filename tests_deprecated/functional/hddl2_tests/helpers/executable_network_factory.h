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
#include <fstream>
#include "models/model_loader.h"

struct ModelParams {
    bool operator==(const ModelParams& rhs) const {
        return inputPrecision == rhs.inputPrecision && inputLayout == rhs.inputLayout;
    }
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NHWC;
};

struct CachedModel {
    ModelParams modelParams;
    std::string graphBlob;
};

class ExecutableNetworkFactory final {
public:
    /**
     * @brief Create executable network based on model name. If model was already compiled in current execution, cached blob will be used
     * @param modelRelativePath relative path of model IR starting from test model path
     *  (example: /KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8)
     */
    static InferenceEngine::ExecutableNetwork createExecutableNetwork(const std::string& modelRelativePath,
                                                                      const ModelParams& modelParams = {});
    /**
     * @brief Load network and modify according to input params
     */
    static InferenceEngine::CNNNetwork createCNNNetwork(const std::string& modelRelativePath,
                                                        const ModelParams& modelParams = {});

    /**
     * @brief Compile and return graph blob, which can be used in ImportMethod
     * @details In already compiled, use temp file
     */
    static std::istringstream getGraphBlob(const std::string& modelRelativePath, const ModelParams& modelParams= {});

private:
    static std::map<std::string, CachedModel> cachedModels;
};
