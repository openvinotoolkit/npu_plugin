//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <fstream>
#include <iostream>
#include <sstream>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Regex.h>

#include "vpux/hwtest/test_case_json_parser.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/string_ref.hpp"

using namespace vpux;

namespace {

llvm::json::Object parse2JSON(StringRef jsonString) {
    // Since the string we're parsing may come from a LIT test, strip off
    // trivial '//' comments.  NB The standard regex library requires C++17 in
    // order to anchor newlines, so we use the LLVM implementation instead.
    static llvm::Regex commentRE{"^ *//.*$", llvm::Regex::Newline};

    auto filteredJSON = jsonString.str();
    for (;;) {
        auto replaced = commentRE.sub("", filteredJSON);
        if (filteredJSON == replaced) {
            break;
        }
        filteredJSON = replaced;
    }

    if (filteredJSON.empty()) {
        throw std::runtime_error{"Expected non-empty filtered JSON"};
    }

    llvm::Expected<llvm::json::Value> exp = llvm::json::parse(filteredJSON);

    if (!exp) {
        auto err = exp.takeError();
        throw std::runtime_error{printToString("HWTEST JSON parsing failed: {0}", err)};
    }

    auto json_object = exp->getAsObject();
    if (!json_object) {
        throw std::runtime_error{"Expected to get JSON as an object"};
    }
    return *json_object;
}

static bool isEqual(StringRef a, const char* b) {
    if (a.size() != strlen(b)) {
        return false;
    }
    auto predicate = [](char left, char right) -> bool {
        return std::tolower(left) == std::tolower(right);
    };
    return std::equal(a.begin(), a.end(), b, predicate);
}

}  // namespace

nb::DType nb::to_dtype(StringRef str) {
    if (isEqual(str, "uint8"))
        return nb::DType::U8;
    if (isEqual(str, "uint4"))
        return nb::DType::U4;
    if (isEqual(str, "int4"))
        return nb::DType::I4;
    if (isEqual(str, "int8"))
        return nb::DType::I8;
    if (isEqual(str, "int32"))
        return nb::DType::I32;
    if (isEqual(str, "fp8"))
        return nb::DType::FP8;
    if (isEqual(str, "fp16"))
        return nb::DType::FP16;
    if (isEqual(str, "fp32"))
        return nb::DType::FP32;
    if (isEqual(str, "bfloat16"))
        return nb::DType::BF16;

    return nb::DType::UNK;
}

std::string nb::to_string(nb::DType dtype) {
    switch (dtype) {
    case nb::DType::U8:
        return "uint8";
    case nb::DType::U4:
        return "uint4";
    case nb::DType::I4:
        return "int4";
    case nb::DType::I8:
        return "int8";
    case nb::DType::I32:
        return "int32";
    case nb::DType::FP8:
        return "fp8";
    case nb::DType::FP16:
        return "fp16";
    case nb::DType::FP32:
        return "fp32";
    case nb::DType::BF16:
        return "bfloat16";
    default:
        return "UNK";
    }
}

MVCNN::Permutation nb::to_odu_permutation(StringRef str) {
    if (isEqual(str, "NHWC"))
        return MVCNN::Permutation::Permutation_ZXY;
    if (isEqual(str, "NWHC"))
        return MVCNN::Permutation::Permutation_ZYX;
    if (isEqual(str, "NWCH"))
        return MVCNN::Permutation::Permutation_YZX;
    if (isEqual(str, "NCWH"))
        return MVCNN::Permutation::Permutation_YXZ;
    if (isEqual(str, "NHCW"))
        return MVCNN::Permutation::Permutation_XZY;
    if (isEqual(str, "NCHW"))
        return MVCNN::Permutation::Permutation_XYZ;
    throw std::runtime_error("ODUPermutation value not supported: " + str.str());

    return MVCNN::Permutation::Permutation_MIN;
}

nb::MemoryLocation nb::to_memory_location(StringRef str) {
    if (isEqual(str, "CMX0")) {
        return nb::MemoryLocation::CMX0;
    }
    if (isEqual(str, "CMX1")) {
        return nb::MemoryLocation::CMX1;
    }
    if (isEqual(str, "DDR")) {
        return nb::MemoryLocation::DDR;
    }

    return nb::MemoryLocation::Unknown;
}

std::string nb::to_string(nb::MemoryLocation memoryLocation) {
    switch (memoryLocation) {
    case MemoryLocation::CMX0:
        return "CMX0";
    case MemoryLocation::CMX1:
        return "CMX1";
    case MemoryLocation::DDR:
        return "DDR";
    default:
        return "Unknown";
    }
}

nb::ActivationType nb::to_activation_type(StringRef str) {
    if (!str.size() || isEqual(str, "None")) {
        return nb::ActivationType::None;
    }
    if (isEqual(str, "LeakyReLU") || isEqual(str, "PReLU")) {
        return nb::ActivationType::LeakyReLU;
    }
    if (isEqual(str, "ReLU")) {
        return nb::ActivationType::ReLU;
    }
    if (isEqual(str, "ReLUX")) {
        return nb::ActivationType::ReLUX;
    }
    if (isEqual(str, "Mish")) {
        return nb::ActivationType::Mish;
    }
    if (isEqual(str, "HSwish")) {
        return nb::ActivationType::HSwish;
    }
    if (isEqual(str, "Sigmoid")) {
        return nb::ActivationType::Sigmoid;
    }
    if (isEqual(str, "Softmax")) {
        return nb::ActivationType::Softmax;
    }
    if (isEqual(str, "vau_sigm")) {
        return nb::ActivationType::vau_sigm;
    }
    if (isEqual(str, "vau_sqrt")) {
        return nb::ActivationType::vau_sqrt;
    }
    if (isEqual(str, "vau_tanh")) {
        return nb::ActivationType::vau_tanh;
    }
    if (isEqual(str, "vau_log")) {
        return nb::ActivationType::vau_log;
    }
    if (isEqual(str, "vau_exp")) {
        return nb::ActivationType::vau_exp;
    }
    if (isEqual(str, "lsu_b16")) {
        return nb::ActivationType::lsu_b16;
    }
    if (isEqual(str, "lsu_b16_vec")) {
        return nb::ActivationType::lsu_b16_vec;
    }
    if (isEqual(str, "sau_dp4")) {
        return nb::ActivationType::sau_dp4;
    }
    if (isEqual(str, "sau_dp4a")) {
        return nb::ActivationType::sau_dp4a;
    }
    if (isEqual(str, "sau_dp4m")) {
        return nb::ActivationType::sau_dp4m;
    }
    if (isEqual(str, "vau_dp4")) {
        return nb::ActivationType::vau_dp4;
    }
    if (isEqual(str, "vau_dp4a")) {
        return nb::ActivationType::vau_dp4a;
    }
    if (isEqual(str, "vau_dp4m")) {
        return nb::ActivationType::vau_dp4m;
    }
    return nb::ActivationType::Unknown;
}

std::string nb::to_string(nb::ActivationType activationType) {
    switch (activationType) {
    case ActivationType::None:
        return "None";
    case ActivationType::ReLU:
        return "ReLU";
    case ActivationType::ReLUX:
        return "ReLUX";
    case ActivationType::LeakyReLU:
        return "LeakyReLU";
    case ActivationType::Mish:
        return "Mish";
    case ActivationType::HSwish:
        return "HSwish";
    case ActivationType::Sigmoid:
        return "Sigmoid";
    case ActivationType::Softmax:
        return "Softmax";
    case ActivationType::vau_sigm:
        return "vau_sigm";
    case ActivationType::vau_sqrt:
        return "vau_sqrt";
    case ActivationType::vau_tanh:
        return "vau_tanh";
    case ActivationType::vau_log:
        return "vau_log";
    case ActivationType::vau_exp:
        return "vau_exp";
    case ActivationType::lsu_b16:
        return "lsu_b16";
    case ActivationType::lsu_b16_vec:
        return "lsu_b16_vec";
    case ActivationType::sau_dp4:
        return "sau_dp4";
    case ActivationType::sau_dp4a:
        return "sau_dp4a";
    case ActivationType::sau_dp4m:
        return "sau_dp4m";
    case ActivationType::vau_dp4:
        return "vau_dp4";
    case ActivationType::vau_dp4a:
        return "vau_dp4a";
    case ActivationType::vau_dp4m:
        return "vau_dp4m";
    default:
        return "Unknown";
    }
}

std::string nb::to_string(CaseType case_) {
    switch (case_) {
    case CaseType::DMA:
        return "DMA";
    case CaseType::ZMajorConvolution:
        return "ZMajorConvolution";
    case CaseType::SparseZMajorConvolution:
        return "SparseZMajorConvolution";
    case CaseType::DepthWiseConv:
        return "DepthWiseConv";
    case CaseType::EltwiseAdd:
        return "EltwiseAdd";
    case CaseType::EltwiseMult:
        return "EltwiseMult";
    case CaseType::MaxPool:
        return "MaxPool";
    case CaseType::AvgPool:
        return "AvgPool";
    case CaseType::DifferentClustersDPU:
        return "DifferentClustersDPU";
    case CaseType::ActShave:
        return "ActShave";
    case CaseType::M2iTask:
        return "M2iTask";
    case CaseType::ReadAfterWriteDPUDMA:
        return "ReadAfterWriteDPUDMA";
    case CaseType::ReadAfterWriteDMADPU:
        return "ReadAfterWriteDMADPU";
    case CaseType::ReadAfterWriteACTDMA:
        return "ReadAfterWriteACTDMA";
    case CaseType::ReadAfterWriteDMAACT:
        return "ReadAfterWriteDMAACT";
    case CaseType::ReadAfterWriteDPUACT:
        return "ReadAfterWriteDPUACT";
    case CaseType::ReadAfterWriteACTDPU:
        return "ReadAfterWriteACTDPU";
    case CaseType::RaceConditionDMA:
        return "RaceConditionDMA";
    case CaseType::RaceConditionDPU:
        return "RaceConditionDPU";
    case CaseType::RaceConditionDPUDMA:
        return "RaceConditionDPUDMA";
    case CaseType::RaceConditionDPUDMAACT:
        return "RaceConditionDPUDMAACT";
    case CaseType::RaceCondition:
        return "RaceCondition";
    default:
        return "unknown";
    }
}

nb::CaseType nb::to_case(StringRef str) {
    if (isEqual(str, "DMA"))
        return CaseType::DMA;
    if (isEqual(str, "ZMajorConvolution"))
        return CaseType::ZMajorConvolution;
    if (isEqual(str, "SparseZMajorConvolution"))
        return CaseType::SparseZMajorConvolution;
    if (isEqual(str, "DepthWiseConv"))
        return CaseType::DepthWiseConv;
    if (isEqual(str, "EltwiseAdd"))
        return CaseType::EltwiseAdd;
    if (isEqual(str, "EltwiseMult"))
        return CaseType::EltwiseMult;
    if (isEqual(str, "MaxPool"))
        return CaseType::MaxPool;
    if (isEqual(str, "AvgPool"))
        return CaseType::AvgPool;
    if (isEqual(str, "DifferentClustersDPU"))
        return CaseType::DifferentClustersDPU;
    if (isEqual(str, "MultiClustersDPU"))
        return CaseType::MultiClustersDPU;
    if (isEqual(str, "ActShave"))
        return CaseType::ActShave;
    if (isEqual(str, "M2iTask"))
        return CaseType::M2iTask;
    if (isEqual(str, "ReadAfterWriteDPUDMA"))
        return CaseType::ReadAfterWriteDPUDMA;
    if (isEqual(str, "ReadAfterWriteDMADPU"))
        return CaseType::ReadAfterWriteDMADPU;
    if (isEqual(str, "ReadAfterWriteACTDMA"))
        return CaseType::ReadAfterWriteACTDMA;
    if (isEqual(str, "ReadAfterWriteDPUACT"))
        return CaseType::ReadAfterWriteDPUACT;
    if (isEqual(str, "ReadAfterWriteACTDPU"))
        return CaseType::ReadAfterWriteACTDPU;
    if (isEqual(str, "ReadAfterWriteDMAACT"))
        return CaseType::ReadAfterWriteDMAACT;
    if (isEqual(str, "RaceConditionDMA"))
        return CaseType::RaceConditionDMA;
    if (isEqual(str, "RaceConditionDPU"))
        return CaseType::RaceConditionDPU;
    if (isEqual(str, "RaceConditionDPUDMA"))
        return CaseType::RaceConditionDPUDMA;
    if (isEqual(str, "RaceConditionDPUDMAACT"))
        return CaseType::RaceConditionDPUDMAACT;
    if (isEqual(str, "RaceCondition"))
        return CaseType::RaceCondition;
    return CaseType::Unknown;
};

nb::M2iFmt nb::to_m2i_fmt(StringRef str) {
    if (isEqual(str, "SP_NV12_8"))
        return nb::M2iFmt::SP_NV12_8;
    if (isEqual(str, "PL_YUV420_8"))
        return nb::M2iFmt::PL_YUV420_8;
    if (isEqual(str, "IL_RGB888"))
        return nb::M2iFmt::IL_RGB888;
    if (isEqual(str, "IL_BGR888"))
        return nb::M2iFmt::IL_BGR888;
    if (isEqual(str, "PL_RGB24"))
        return nb::M2iFmt::PL_RGB24;
    if (isEqual(str, "PL_BGR24"))
        return nb::M2iFmt::PL_BGR24;
    if (isEqual(str, "PL_FP16_RGB"))
        return nb::M2iFmt::PL_FP16_RGB;
    if (isEqual(str, "PL_FP16_BGR"))
        return nb::M2iFmt::PL_FP16_BGR;
    return M2iFmt::Unknown;
}

nb::QuantParams nb::TestCaseJsonDescriptor::loadQuantizationParams(llvm::json::Object* obj) {
    nb::QuantParams result;
    auto* qp = obj->getObject("quantization");
    if (qp) {
        result.present = true;
        result.scale = qp->getNumber("scale").getValue();
        result.zeropoint = qp->getInteger("zeropoint").getValue();
        result.low_range = static_cast<std::int64_t>(qp->getNumber("low_range").getValue());
        result.high_range = static_cast<std::int64_t>(qp->getNumber("high_range").getValue());
    }
    return result;
}

nb::RaceConditionParams nb::TestCaseJsonDescriptor::loadRaceConditionParams(llvm::json::Object* jsonObj) {
    nb::RaceConditionParams params;
    params.iterationsCount = jsonObj->getInteger("iteration_count").getValue();
    params.requestedClusters = jsonObj->getInteger("requested_clusters").getValue();
    params.requestedUnits = jsonObj->getInteger("requested_units").getValue();

    return params;
}

nb::DPUTaskParams nb::TestCaseJsonDescriptor::loadDPUTaskParams(llvm::json::Object* jsonObj) {
    nb::DPUTaskParams params;
    auto* taskParams = jsonObj->getObject("DPUTaskParams");
    const auto* jsonOutClusters = taskParams->getArray("output_cluster");

    VPUX_THROW_UNLESS(jsonOutClusters != nullptr, "loadDPUTaskParams: cannot find output_cluster config param");

    params.outputClusters.resize(jsonOutClusters->size());
    for (size_t i = 0; i < jsonOutClusters->size(); i++) {
        params.outputClusters[i] = (*jsonOutClusters)[i].getAsInteger().getValue();
    }

    params.inputCluster = taskParams->getInteger("input_cluster").getValue();
    params.weightsCluster = taskParams->getInteger("weights_cluster").getValue();
    params.weightsTableCluster = taskParams->getInteger("weights_table_cluster").getValue();

    return params;
}

nb::MultiClusterDPUParams nb::TestCaseJsonDescriptor::loadMultiClusterDPUParams(llvm::json::Object* jsonObj) {
    nb::MultiClusterDPUParams params;
    auto* taskParams = jsonObj->getObject("DPUTaskParams");

    const auto* jsonTaskClusters = taskParams->getArray("task_clusters");
    VPUX_THROW_UNLESS(jsonTaskClusters != nullptr, "loadMultiClusterDPUParams: cannot find task_clusters config param");

    params.taskClusters.resize(jsonTaskClusters->size());
    for (size_t i = 0; i < jsonTaskClusters->size(); i++) {
        params.taskClusters[i] = (*jsonTaskClusters)[i].getAsInteger().getValue();
    }

    const std::unordered_map<llvm::StringRef, SegmentationType> segmentOptions = {{"SOK", SegmentationType::SOK},
                                                                                  {"SOH", SegmentationType::SOH}};

    auto segmentation = taskParams->getString("segmentation");
    VPUX_THROW_UNLESS(segmentation.hasValue() && segmentOptions.find(segmentation.getValue()) != segmentOptions.end(),
                      "loadMultiClusterDPUParams: failed to get valid segmentation type");

    params.segmentation = segmentOptions.at(segmentation.getValue().str());
    params.broadcast = taskParams->getBoolean("broadcast").getValue();

    return params;
}

SmallVector<nb::InputLayer> nb::TestCaseJsonDescriptor::loadInputLayer(llvm::json::Object* jsonObj) {
    SmallVector<nb::InputLayer> result;

    auto* inputArray = jsonObj->getArray("input");
    if (!inputArray) {
        return result;
    }

    result.resize(inputArray->size());
    for (size_t inIdx = 0; inIdx < inputArray->size(); inIdx++) {
        auto inputObj = (*inputArray)[inIdx].getAsObject();
        auto* shape = inputObj->getArray("shape");
        VPUX_THROW_UNLESS(shape != nullptr, "loadInputLayer: missing shape");

        for (size_t i = 0; i < shape->size(); i++) {
            result[inIdx].shape[i] = (*shape)[i].getAsInteger().getValue();
        }

        result[inIdx].qp = loadQuantizationParams(inputObj);
        result[inIdx].dtype = to_dtype(inputObj->getString("dtype").getValue().str());
    }

    return result;
}

nb::WeightLayer nb::TestCaseJsonDescriptor::loadWeightLayer(llvm::json::Object* jsonObj) {
    nb::WeightLayer result;

    std::string layerType = "weight";

    auto* weight = jsonObj->getObject(layerType);
    if (!weight) {
        return result;
    }

    auto* shape = weight->getArray("shape");
    VPUX_THROW_UNLESS(shape != nullptr, "loadWeightLayer: missing shape");

    for (size_t i = 0; i < shape->size(); i++) {
        result.shape[i] = (*shape)[i].getAsInteger().getValue();
    }

    result.qp = loadQuantizationParams(weight);
    result.dtype = to_dtype(weight->getString("dtype").getValue().str());

    auto filename = weight->getString("file_path");
    if (filename) {
        result.filename = filename.getValue().str();
    }

    return result;
}

SmallVector<nb::OutputLayer> nb::TestCaseJsonDescriptor::loadOutputLayer(llvm::json::Object* jsonObj) {
    SmallVector<nb::OutputLayer> result;

    auto* outputArray = jsonObj->getArray("output");
    if (!outputArray) {
        return result;
    }

    result.resize(outputArray->size());
    for (size_t outIdx = 0; outIdx < outputArray->size(); outIdx++) {
        auto outputObj = (*outputArray)[outIdx].getAsObject();
        auto* shape = outputObj->getArray("shape");
        VPUX_THROW_UNLESS(shape != nullptr, "loadOutputLayer: missing shape");

        for (size_t i = 0; i < shape->size(); i++) {
            result[outIdx].shape[i] = (*shape)[i].getAsInteger().getValue();
        }

        result[outIdx].qp = loadQuantizationParams(outputObj);
        result[outIdx].dtype = to_dtype(outputObj->getString("dtype").getValue().str());
    }

    return result;
}

nb::DMAparams nb::TestCaseJsonDescriptor::loadDMAParams(llvm::json::Object* jsonObj) {
    nb::DMAparams result;

    auto* params = jsonObj->getObject("DMA_params");
    if (!params) {
        VPUX_THROW("DMA params doesn't provided");
    }

    auto srcMemLoc = params->getString("src_memory_location");
    VPUX_THROW_UNLESS(srcMemLoc.hasValue(), "Source memory location doesn't provided");
    result.srcLocation = to_memory_location(srcMemLoc.getValue());

    auto dstMemLoc = params->getString("dst_memory_location");
    VPUX_THROW_UNLESS(dstMemLoc.hasValue(), "Destination memory location doesn't provided");
    result.dstLocation = to_memory_location(dstMemLoc.getValue());

    auto dmaEngine = params->getInteger("dma_engine");
    VPUX_THROW_UNLESS(dmaEngine.hasValue(), "DMA engine doesn't provided");
    result.engine = dmaEngine.getValue();

    return result;
}

nb::M2iLayer nb::TestCaseJsonDescriptor::loadM2iLayer(llvm::json::Object* jsonObj) {
    nb::M2iLayer result;

    auto* params = jsonObj->getObject("m2i_params");
    if (!params) {
        VPUX_THROW("M2I params not provided");
    }
    const auto inputFmt = params->getString("input_fmt");
    const auto outputFmt = params->getString("output_fmt");
    const auto cscFlag = params->getBoolean("do_csc");
    const auto normFlag = params->getBoolean("do_norm");

    VPUX_THROW_UNLESS(inputFmt.hasValue(), "input_fmt not provided !");
    VPUX_THROW_UNLESS(outputFmt.hasValue(), "output_fmt not provided !");
    VPUX_THROW_UNLESS(cscFlag.hasValue(), "do_csc not provided !");
    VPUX_THROW_UNLESS(normFlag.hasValue(), "do_norm not provided !");

    result.iFmt = to_m2i_fmt(inputFmt.getValue());  // str to enum
    result.oFmt = to_m2i_fmt(outputFmt.getValue());
    result.doCsc = cscFlag.getValue();
    result.doNorm = normFlag.getValue();

    // Optional params for RESIZE and NORM
    const auto* sizesVec = params->getArray("output_sizes");
    const auto* coefsVec = params->getArray("norm_coefs");
    VPUX_THROW_UNLESS(sizesVec != nullptr, "loadM2iLayer: missing sizesVec");
    VPUX_THROW_UNLESS(coefsVec != nullptr, "loadM2iLayer: missing coefsVec");

    for (size_t i = 0; i < sizesVec->size(); i++) {
        auto elem = (*sizesVec)[i].getAsInteger();
        if (elem.hasValue()) {
            result.outSizes.push_back(static_cast<int>(elem.getValue()));
        }
    }

    for (size_t i = 0; i < coefsVec->size(); i++) {
        auto elem = (*coefsVec)[i].getAsNumber();  // double
        if (elem.hasValue()) {
            result.normCoefs.push_back(static_cast<float>(elem.getValue()));
        }
    }

    return result;
}

nb::ConvLayer nb::TestCaseJsonDescriptor::loadConvLayer(llvm::json::Object* jsonObj) {
    std::string layerType = "conv_op";

    nb::ConvLayer result;

    auto* op = jsonObj->getObject("conv_op");
    VPUX_THROW_UNLESS(op != nullptr, "loadConvLayer: missing conv_op config");

    auto* strides = op->getArray("stride");
    VPUX_THROW_UNLESS(strides != nullptr, "loadConvLayer: missing strides");

    for (size_t i = 0; i < strides->size(); i++) {
        auto stride = (*strides)[i].getAsInteger();
        if (stride.hasValue()) {
            result.stride.at(i) = stride.getValue();
        }
    }

    auto* pads = op->getArray("pad");
    VPUX_THROW_UNLESS(pads != nullptr, "loadConvLayer: missing pads");

    for (size_t i = 0; i < pads->size(); i++) {
        auto pad = (*pads)[i].getAsInteger();
        if (pad.hasValue()) {
            result.pad.at(i) = pad.getValue();
        }
    }

    result.group = op->getInteger("group").getValue();
    result.dilation = op->getInteger("dilation").getValue();
    auto compress = op->getInteger("compress");
    if (compress.hasValue()) {
        result.compress = (compress.getValue() > 0);
    } else {
        result.compress = false;
    }

    auto mpe_cub = op->getString("mpe_cub");
    if (mpe_cub.hasValue()) {
        if (mpe_cub.getValue() == "CUBOID_8x16") {
            result.cube_mode = vpux::VPU::MPEMode::CUBOID_8x16;
        } else if (mpe_cub.getValue() == "CUBOID_4x16") {
            result.cube_mode = vpux::VPU::MPEMode::CUBOID_4x16;
        }
        // TODO: Check for the default (CUBOID_16x16) and log if it's something else.
    }

    return result;
}

nb::PoolLayer nb::TestCaseJsonDescriptor::loadPoolLayer(llvm::json::Object* jsonObj) {
    std::string layerType = "pool_op";

    nb::PoolLayer result;

    auto* op = jsonObj->getObject("pool_op");
    VPUX_THROW_UNLESS(op != nullptr, "loadPoolLayer: missing pool_op config");

    auto* kernel_shape = op->getArray("kernel_shape");
    VPUX_THROW_UNLESS(kernel_shape != nullptr, "loadPoolLayer: missing kernel_shape");

    for (size_t i = 0; i < kernel_shape->size(); i++) {
        auto kernelsize = (*kernel_shape)[i].getAsInteger();
        if (kernelsize.hasValue()) {
            result.kernel_shape.at(i) = kernelsize.getValue();
        }
    }
    auto* strides = op->getArray("stride");
    VPUX_THROW_UNLESS(strides != nullptr, "loadPoolLayer: missing stride");

    for (size_t i = 0; i < strides->size(); i++) {
        auto stride = (*strides)[i].getAsInteger();
        if (stride.hasValue()) {
            result.stride.at(i) = stride.getValue();
        }
    }

    auto* pads = op->getArray("pad");
    if (!pads) {
        return result;
    }
    for (size_t i = 0; i < pads->size(); i++) {
        auto pad = (*pads)[i].getAsInteger();
        if (pad.hasValue()) {
            result.pad.at(i) = pad.getValue();
        }
    }

    return result;
}

nb::ActivationLayer nb::TestCaseJsonDescriptor::loadActivationLayer(llvm::json::Object* jsonObj) {
    nb::ActivationLayer result;

    auto* act = jsonObj->getObject("activation");
    if (!act) {
        // This is fine; just return a default activation layer.
        return result;
    }

    result.activationType = to_activation_type(act->getString("name").getValue().str());

    auto alpha = act->getNumber("alpha");
    if (alpha.hasValue()) {
        result.alpha = alpha.getValue();
    }

    auto maximum = act->getNumber("max");
    if (maximum.hasValue()) {
        result.maximum = maximum.getValue();
    }

    auto axis = act->getNumber("axis");
    if (axis.hasValue()) {
        result.axis = vpux::checked_cast<size_t>(axis.getValue());
    }

    return result;
}

std::size_t nb::TestCaseJsonDescriptor::loadIterationCount(llvm::json::Object* jsonObj) {
    return jsonObj->getInteger("iteration_count").getValue();
}

std::size_t nb::TestCaseJsonDescriptor::loadClusterNumber(llvm::json::Object* jsonObj) {
    return jsonObj->getInteger("cluster_number").getValue();
}

nb::TestCaseJsonDescriptor::TestCaseJsonDescriptor(StringRef jsonString) {
    if (!jsonString.empty()) {
        parse(parse2JSON(jsonString));
    }
}

nb::TestCaseJsonDescriptor::TestCaseJsonDescriptor(llvm::json::Object jsonObject) {
    parse(jsonObject);
}

void nb::TestCaseJsonDescriptor::parse(llvm::json::Object json_obj) {
    auto architecture = json_obj.getString("architecture");
    if (!architecture) {
        throw std::runtime_error{"Failed to get architecture"};
    }
    const auto architectureSymbol = vpux::VPU::symbolizeArchKind(architecture.getValue());
    if (!architectureSymbol.hasValue()) {
        throw std::runtime_error{"Failed to parse architecture"};
    }
    architecture_ = architectureSymbol.getValue();

    auto case_type = json_obj.getString("case_type");
    if (!case_type) {
        throw std::runtime_error{"Failed to get case type"};
    }

    caseType_ = nb::to_case(case_type.getValue());
    caseTypeStr_ = case_type.getValue().str();
    inLayers_ = loadInputLayer(&json_obj);
    outLayers_ = loadOutputLayer(&json_obj);
    activationLayer_ = loadActivationLayer(&json_obj);

    // Load conv json attribute values. Similar implementation for ALL HW layers (DW, group conv, Av/Max pooling and
    // eltwise needed).
    if (caseType_ == CaseType::DMA) {
        DMAparams_ = loadDMAParams(&json_obj);
        return;
    }

    if (caseType_ == CaseType::ZMajorConvolution || caseType_ == CaseType::DepthWiseConv ||
        caseType_ == CaseType::RaceConditionDPU || caseType_ == CaseType::RaceConditionDPUDMA ||
        caseType_ == CaseType::RaceConditionDPUDMAACT || caseType_ == CaseType::ReadAfterWriteDPUDMA ||
        caseType_ == CaseType::ReadAfterWriteDMADPU || caseType_ == CaseType::ReadAfterWriteDPUACT ||
        caseType_ == CaseType::ReadAfterWriteACTDPU || caseType_ == CaseType::SparseZMajorConvolution) {
        wtLayer_ = loadWeightLayer(&json_obj);
        convLayer_ = loadConvLayer(&json_obj);

        if (caseType_ == CaseType::ZMajorConvolution) {
            odu_permutation_ = to_odu_permutation(json_obj.getString("output_order").getValue());
        }

        if (caseType_ == CaseType::RaceConditionDPU || caseType_ == CaseType::RaceConditionDPUDMA ||
            caseType_ == CaseType::ReadAfterWriteDPUDMA || caseType_ == CaseType::ReadAfterWriteDMADPU) {
            iterationCount_ = loadIterationCount(&json_obj);
        }
        if (caseType_ == CaseType::RaceConditionDPUDMAACT || caseType_ == CaseType::ReadAfterWriteDPUACT ||
            caseType_ == CaseType::ReadAfterWriteACTDPU) {
            iterationCount_ = loadIterationCount(&json_obj);
            activationLayer_ = loadActivationLayer(&json_obj);
        }
        if (caseType_ == CaseType::ReadAfterWriteDPUDMA || caseType_ == CaseType::ReadAfterWriteDMADPU ||
            caseType_ == CaseType::ReadAfterWriteDPUACT || caseType_ == CaseType::ReadAfterWriteACTDPU) {
            clusterNumber_ = loadClusterNumber(&json_obj);
        }
        return;
    }

    if (caseType_ == CaseType::DifferentClustersDPU) {
        wtLayer_ = loadWeightLayer(&json_obj);
        convLayer_ = loadConvLayer(&json_obj);
        DPUTaskParams_ = loadDPUTaskParams(&json_obj);
        return;
    }

    if (caseType_ == CaseType::MultiClustersDPU) {
        wtLayer_ = loadWeightLayer(&json_obj);
        convLayer_ = loadConvLayer(&json_obj);
        multiClusterDPUParams_ = loadMultiClusterDPUParams(&json_obj);
        return;
    }

    if (caseType_ == CaseType::EltwiseAdd || caseType_ == CaseType::EltwiseMult) {
        wtLayer_ = loadWeightLayer(&json_obj);
        return;
    }

    if (caseType_ == CaseType::MaxPool || caseType_ == CaseType::AvgPool) {
        poolLayer_ = loadPoolLayer(&json_obj);
        return;
    }

    if (caseType_ == CaseType::ActShave) {
        return;
    }

    if (caseType_ == CaseType::ReadAfterWriteACTDMA || caseType_ == CaseType::ReadAfterWriteDMAACT) {
        activationLayer_ = loadActivationLayer(&json_obj);
        iterationCount_ = loadIterationCount(&json_obj);
        clusterNumber_ = loadClusterNumber(&json_obj);
        return;
    }

    if (caseType_ == CaseType::RaceConditionDMA) {
        iterationCount_ = loadIterationCount(&json_obj);
        return;
    }

    if (caseType_ == CaseType::RaceCondition) {
        if (auto underlyingOp = json_obj.getObject("operation")) {
            this->underlyingOp_ = std::make_shared<TestCaseJsonDescriptor>(*underlyingOp);
            raceConditionParams_ = loadRaceConditionParams(&json_obj);
            return;
        }
    }

    if (caseType_ == CaseType::M2iTask) {
        m2iLayer_ = loadM2iLayer(&json_obj);
        return;
    }

    throw std::runtime_error{printToString("Unsupported case type: {0}", caseTypeStr_)};
}

nb::CaseType nb::TestCaseJsonDescriptor::loadCaseType(llvm::json::Object* jsonObj) {
    return to_case(jsonObj->getString("case_type").getValue().str());
}
