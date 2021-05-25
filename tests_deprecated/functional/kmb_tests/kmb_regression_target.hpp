//
// Copyright 2019 Intel Corporation.
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
