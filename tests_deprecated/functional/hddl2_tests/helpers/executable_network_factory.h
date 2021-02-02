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
