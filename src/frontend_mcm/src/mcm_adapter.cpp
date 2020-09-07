//
// Copyright 2019-2020 Intel Corporation.
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

#include "mcm_adapter.hpp"

#include "frontend_mcm.hpp"

#ifdef ENABLE_MCM_COMPILER

#include <file_utils.h>
#include <net_pass.h>
#include <sys/stat.h>

#include <ie_icnn_network.hpp>
#include <ie_itt.hpp>
#include <ie_util_internal.hpp>

#include "include/mcm/compiler/compilation_unit.hpp"
#endif

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile/graphfile_generated.h>

#include "converters.hpp"
#include "ie_memcpy.h"

using namespace InferenceEngine;
using namespace vpu;

#ifdef ENABLE_MCM_COMPILER

static std::string getMcmLogLevel(LogLevel lvl) {
    switch (lvl) {
    case LogLevel::None:
        return "Silent";

    case LogLevel::Fatal:
    case LogLevel::Error:
        return "Error";

    case LogLevel::Warning:
        return "Warning";

    case LogLevel::Info:
        return "Info";

    case LogLevel::Debug:
    case LogLevel::Trace:
        return "Debug";

    default:
        return "Silent";
    }
}

static std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(
    const std::string& tensorName, const InferenceEngine::TensorDesc& tensorInfo) {
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild =
        std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    toBuild->name = tensorName;
    const InferenceEngine::SizeVector& dimVec = tensorInfo.getDims();
    for (const size_t& dim : dimVec) {
        toBuild->dimensions.push_back(dim);
    }
    toBuild->strides = layoutToOrder(tensorInfo.getLayout());
    toBuild->data_dtype = precisionToDType(tensorInfo.getPrecision());
    toBuild->data = nullptr;

    return toBuild;
}

std::vector<char> serializeMetaData(const char* memBlobData, const InferenceEngine::InputsDataMap& inputInfo,
    const InferenceEngine::OutputsDataMap& outputInfo) {
    const MVCNN::GraphFile* graphFilePtr = MVCNN::GetGraphFile(memBlobData);
    MVCNN::GraphFileT graphFileInstance;
    graphFilePtr->UnPackTo(&graphFileInstance);

    for (auto inIter = inputInfo.begin(); inIter != inputInfo.end(); ++inIter) {
        graphFileInstance.header->in_tensor_desc.push_back(
            buildTensorReference(inIter->first, inIter->second->getTensorDesc()));
    }

    for (auto outIter = outputInfo.begin(); outIter != outputInfo.end(); ++outIter) {
        graphFileInstance.header->out_tensor_desc.push_back(
            buildTensorReference(outIter->first, outIter->second->getTensorDesc()));
    }

    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<MVCNN::GraphFile> offset = MVCNN::CreateGraphFile(builder, &graphFileInstance);
    MVCNN::FinishGraphFileBuffer(builder, offset);
    std::vector<char> binaryData(builder.GetSize());
    ie_memcpy(binaryData.data(), binaryData.size(), builder.GetBufferPointer(), binaryData.size());

    return binaryData;
}

void MCMAdapter::compileNetwork(
    InferenceEngine::ICNNNetwork& network, const MCMConfig& config, std::vector<char>& blob) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "compileNetwork");
    auto unit = std::make_shared<mv::CompilationUnit>(network.getName());

    if (unit == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.";
    }
    bool ti_proc_ok =
        !InferenceEngine::NetPass::CombineRNNSeq(network) ? InferenceEngine::NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

    blob.clear();
    Logger::Ptr _logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    mv::OpModel& modelMcm = unit->model();

    auto frontEnd = std::make_shared<FrontEndMcm>(modelMcm, config);

    const auto& targetName = config.mcmTargetDesciptor();
    const auto targetPath =
        getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";

    // TODO: This hack needs to be fixed
    auto compDescName = config.mcmCompilationDesciptor();
    ie::InputsDataMap networkInputs;
    bool layoutNCHW = true;
    network.getInputsInfo(networkInputs);
    for (const auto& netInput : networkInputs) {
        if (netInput.second->getLayout() != InferenceEngine::Layout::NCHW) {
            layoutNCHW = false;
            break;
        }
    }
    if (layoutNCHW) {
        compDescName = "release_kmb_with_CM_Conv";
    }

    const auto compDescPath =
        getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" + compDescName + ".json";

    const auto& resultsPath = config.mcmCompilationResultsPath();
    auto resultNames = config.mcmCompilationResults();
    resultNames = (resultNames == "") ? network.getName() : resultNames;

    mkdir(resultsPath.c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName).c_str(), 0755);
    mkdir((resultsPath + "/" + compDescName + "/" + targetName).c_str(), 0755);

    std::string resultsFullName = resultsPath + "/" + compDescName + "/" + targetName + "/" + resultNames;

    if (!unit->loadTargetDescriptor(targetPath)) {
        THROW_IE_EXCEPTION << "Target description loading failed! Path: " << targetPath;
    }

    if (!unit->loadCompilationDescriptor(compDescPath)) {
        THROW_IE_EXCEPTION << "Compilation description loading failed! Path: " << compDescPath;
    };
    auto& compDesc = unit->compilationDescriptor();

    // parse model after loading config
    frontEnd->buildInitialModel(network);
    if (config.mcmParseOnly()) {
        return;
    }

    _logger->info("Path for results: %s (%s)", resultsFullName, std::strerror(errno));

    if (config.mcmGenerateBlob()) {
        //-----------------------------------------------------------------------
        // Note: There are different passes in different Compilation Descriptors
        // Just now we try two of them
        //-----------------------------------------------------------------------
        bool isBlobFileSet = false;

        if (compDesc.validPass("GenerateBlobKmb")) {
            // validPass returns true here but 'setPassArg' attempt causes
            // 'Trying to set arguments for a non-existent pass' error
            // so we use try catch
            try {
                compDesc.setPassArg("GenerateBlobKmb", "output", resultsFullName + ".blob");
                isBlobFileSet = true;
            } catch (...) {
            }
        }
        if (!isBlobFileSet) {
            VPU_THROW_EXCEPTION << "Can't set mcmCompiler arguments for blob generation!";
        }
    }

    if (config.mcmGenerateDOT()) {
        VPU_THROW_EXCEPTION << "Not implemented";
    }

    compDesc.setPassArg("GlobalConfigParams", "verbose", getMcmLogLevel(config.mcmLogLevel()));
    compDesc.setPassArg("GlobalConfigParams", "ScaleFuseInput", config.scaleFuseInput());

    if (!config.mcmCompilationPassBanList().empty()) {
        std::stringstream banList{config.mcmCompilationPassBanList()};
        std::string groupPassPair;
        while (std::getline(banList, groupPassPair, ';')) {
            const auto delim = groupPassPair.find(',');
            VPU_THROW_UNLESS(delim != std::string::npos,
                "McmCompilationPassBanList parsing error: provided value '%s'"
                "should have comma separated Group,Pass string",
                groupPassPair);
            const auto group = groupPassPair.substr(0, delim);
            const auto pass = groupPassPair.substr(delim + 1, std::string::npos);
            compDesc.remove(group, pass);
        }
    }

    IE_ASSERT(unit->initialize());

    try {
        auto result = unit->run();

        if (config.mcmGenerateJSON()) {
            std::fstream file_out(resultsFullName + ".json", std::fstream::out);
            file_out << result.toString() << std::endl;
            file_out.close();
        }
    } catch (const std::exception& ex) {
        VPU_THROW_EXCEPTION << "Caught exception during unit run: " << ex.what();
    } catch (...) {
        VPU_THROW_EXCEPTION << "Unknown exception during unit run";
    }

    if (config.mcmGenerateDOT()) {
        rename("original_model.dot", (resultsFullName + "_original.dot").c_str());
        rename("adapt_model.dot", (resultsFullName + "_adapt.dot").c_str());
        rename("final_model.dot", (resultsFullName + ".dot").c_str());
    }

    if (config.mcmGenerateBlob()) {
        InferenceEngine::InputsDataMap inputInfo;
        network.getInputsInfo(inputInfo);

        InferenceEngine::OutputsDataMap outputInfo;
        network.getOutputsInfo(outputInfo);

        auto memBlob = unit->getBlob();
        std::vector<char> binaryData;
        if (memBlob == nullptr) {
            std::ifstream blobFile(resultsFullName + ".blob", std::ios::binary);
            if (blobFile) {
                std::ostringstream blobContentStream;
                blobContentStream << blobFile.rdbuf();
                const std::string& blobContentString = blobContentStream.str();
                binaryData = serializeMetaData(blobContentString.c_str(), inputInfo, outputInfo);
            } else {
                VPU_THROW_EXCEPTION << "Can not open blob file " << resultsFullName + ".blob"
                                    << ". It was not created by mcmCompiler!";
            }
        } else {
            binaryData = serializeMetaData(memBlob->data(), inputInfo, outputInfo);
        }
        std::copy(binaryData.begin(), binaryData.end(), std::back_inserter(blob));

        if (blob.empty()) {
            VPU_THROW_EXCEPTION << "Blob file " << resultsFullName + ".blob"
                                << " created by mcmCompiler is empty!";
        }
    }
}

std::set<std::string> MCMAdapter::getSupportedLayers(InferenceEngine::ICNNNetwork& network, const MCMConfig& config) {
    std::shared_ptr<mv::CompilationUnit> tmpCompiler = std::make_shared<mv::CompilationUnit>(network.getName());
    if (tmpCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.\n"
                           << "Supported format: FP32 and FP16.";
    }

    auto frontEnd = std::make_shared<FrontEndMcm>(tmpCompiler->model(), config);

    return frontEnd->checkSupportedLayers(network);
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() {
    std::shared_ptr<mv::CompilationUnit> tmpCompiler = std::make_shared<mv::CompilationUnit>("testModel");
    return tmpCompiler != nullptr;
}

#else

void vpu::MCMAdapter::compileNetwork(
    InferenceEngine::ICNNNetwork& /*network*/, const MCMConfig& /*config*/, std::vector<char>& /*blob*/) {
    THROW_IE_EXCEPTION << "Compiler is disabled";
}

std::set<std::string> vpu::MCMAdapter::getSupportedLayers(InferenceEngine::ICNNNetwork&, const MCMConfig&) {
    THROW_IE_EXCEPTION << "Compiler is disabled";
}

bool vpu::MCMAdapter::isMCMCompilerAvailable() { return false; }

#endif  // ENABLE_MCM_COMPILER

std::pair<InferenceEngine::InputsDataMap, InferenceEngine::OutputsDataMap> vpu::MCMAdapter::deserializeMetaData(
    const std::vector<char>& outBlob, const MCMConfig& config) {
    Logger::Ptr logger = std::make_shared<Logger>("compileMCM", config.logLevel(), consoleOutput());
    if (logger == nullptr) {
        THROW_IE_EXCEPTION << "Logger has not been created";
    }
    const MVCNN::GraphFile* graphFilePtr = MVCNN::GetGraphFile(outBlob.data());
    MVCNN::GraphFileT graphFileInstance;
    graphFilePtr->UnPackTo(&graphFileInstance);

    InferenceEngine::InputsDataMap resultNetworkInputs;
    size_t inputTensorsCount = graphFileInstance.header->in_tensor_desc.size();
    logger->debug("inputTensorsCount: %d", inputTensorsCount);
    for (size_t inputIdx = 0; inputIdx < inputTensorsCount; inputIdx++) {
        std::unique_ptr<MVCNN::TensorReferenceT>& tensorRef = graphFileInstance.header->in_tensor_desc.at(inputIdx);
        std::ostringstream inputSerializer;
        inputSerializer << "Name: " << tensorRef->name << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions.begin(), tensorRef->dimensions.end(), std::back_inserter(dimVec));
        inputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            inputSerializer << " " << dim << " ";
        }
        inputSerializer << "}" << std::endl;
        InferenceEngine::Layout ieLayout = orderToLayout(tensorRef->strides);
        InferenceEngine::Precision iePrecision = DTypeToPrecision(tensorRef->data_dtype);
        inputSerializer << "Layout: " << ieLayout << std::endl;
        inputSerializer << "Precision: " << iePrecision << std::endl;

        InferenceEngine::TensorDesc inputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data inputData(tensorRef->name, inputDesc);
        logger->debug("input info:\n%s\n", inputSerializer.str());

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        resultNetworkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    }

    InferenceEngine::OutputsDataMap resultNetworkOutputs;
    size_t outputTensorsCount = graphFileInstance.header->out_tensor_desc.size();
    logger->debug("outputTensorsCount: %d", outputTensorsCount);
    for (size_t outputIdx = 0; outputIdx < outputTensorsCount; outputIdx++) {
        std::unique_ptr<MVCNN::TensorReferenceT>& tensorRef = graphFileInstance.header->out_tensor_desc.at(outputIdx);
        std::ostringstream outputSerializer;
        outputSerializer << "Name: " << tensorRef->name << std::endl;
        InferenceEngine::SizeVector dimVec;
        std::copy(tensorRef->dimensions.begin(), tensorRef->dimensions.end(), std::back_inserter(dimVec));
        outputSerializer << "Dims: {";
        for (const size_t& dim : dimVec) {
            outputSerializer << " " << dim << " ";
        }
        outputSerializer << "}" << std::endl;
        InferenceEngine::Layout ieLayout = orderToLayout(tensorRef->strides);
        InferenceEngine::Precision iePrecision = DTypeToPrecision(tensorRef->data_dtype);
        outputSerializer << "Layout: " << ieLayout << std::endl;
        outputSerializer << "Precision: " << iePrecision << std::endl;
        logger->debug("output info:\n%s\n", outputSerializer.str());

        InferenceEngine::TensorDesc outputDesc(iePrecision, dimVec, ieLayout);
        InferenceEngine::Data outputData(tensorRef->name, outputDesc);
        resultNetworkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    }

    return {resultNetworkInputs, resultNetworkOutputs};
}
