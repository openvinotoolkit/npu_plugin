//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

const size_t NUMBER_OF_TOP_CLASSES = 1;
const std::string YOLO_GRAPH_NAME = "tiny-yolo-v2.blob";

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

class VpuInferWithPath : public vpuLayersTests, public testing::WithParamInterface<modelBlobsInfo> {};

#endif

}  // namespace KmbRegressionTarget
