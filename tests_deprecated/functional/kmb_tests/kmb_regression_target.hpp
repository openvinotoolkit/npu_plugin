//
// Copyright 2019 Intel Corporation.
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

#include <string>
#include <vpu_layers_tests.hpp>

namespace KmbRegressionTarget {

struct CompilationParameter {
    CompilationParameter() = default;
    CompilationParameter(std::string pName, std::string pathToNetwork, std::string pathToWeights)
        : name(pName), path_to_network(pathToNetwork), path_to_weights(pathToWeights) {};

    std::string name;
    std::string path_to_network;
    std::string path_to_weights;
};

#if defined(__arm__) || defined(__aarch64__)

const size_t NUMBER_OF_TOP_CLASSES = 5;
const std::string YOLO_GRAPH_NAME = "tiny-yolo-v2.blob";

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

class VpuInferWithPath : public vpuLayersTests, public testing::WithParamInterface<modelBlobsInfo> {};

#endif

}  // namespace KmbRegressionTarget
