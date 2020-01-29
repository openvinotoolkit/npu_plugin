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

#include "hddl2_graph.h"

#include <fstream>
#include <memory>
#include <string>

#include "hddl2_exceptions.h"
#include "hddl2_helpers.h"

using namespace vpu::HDDL2Plugin;
//------------------------------------------------------------------------------
//      class HDDL2Graph Implementation
//------------------------------------------------------------------------------
// TODO Hardcoded resnet until mcm will be supported
InferenceEngine::InputsDataMap HDDL2Graph::getInputsInfo() const noexcept {
    InferenceEngine::InputsDataMap m_networkInputs;
    InferenceEngine::SizeVector inputDims({1, 3, 224, 224});
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    return m_networkInputs;
}

// TODO Hardcoded resnet until mcm will be supported
InferenceEngine::OutputsDataMap HDDL2Graph::getOutputsInfo() const noexcept {
    InferenceEngine::OutputsDataMap m_networkOutputs;
    InferenceEngine::SizeVector outputDims({1, 1024, 1, 1});
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);

    InferenceEngine::Data outputData("output", outputDesc);
    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    return m_networkOutputs;
}

std::string HDDL2Graph::getGraphName() {
    // TODO hardcoded for now
    _graphName = "resNet";
    return _graphName;
}

const std::string& HDDL2Graph::getGraphBlob() const { return _blobContentString; }

//------------------------------------------------------------------------------
//      class HDDL2CompiledGraph Implementation
//------------------------------------------------------------------------------
HDDL2CompiledGraph::HDDL2CompiledGraph(const IE::ICNNNetwork& network) {
    UNUSED(network);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

//------------------------------------------------------------------------------
//      class HDDL2CompiledGraph Implementation
//------------------------------------------------------------------------------
HDDL2ImportedGraph::HDDL2ImportedGraph(const std::string& blobFilename): _blobFileName(blobFilename) {
    std::ifstream blobFile(blobFilename, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << FILES_ERROR_str << "Could not open file: " << blobFilename;
    }
    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();

    _blobContentString = blobContentStream.str();
}
