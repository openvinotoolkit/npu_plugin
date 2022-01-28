//
// Copyright 2021 Intel Corporation.
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

#include <fstream>
#include <iostream>
#include <sstream>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Regex.h>

#include "vpux/hwtest/test_case_json_parser.hpp"

static bool isEqual(llvm::StringRef a, const char* b) {
    if (a.size() != strlen(b)) {
        return false;
    }
    auto predicate = [](char left, char right) -> bool {
        return std::tolower(left) == std::tolower(right);
    };
    return std::equal(a.begin(), a.end(), b, predicate);
}

nb::DType nb::to_dtype(llvm::StringRef str) {
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

MVCNN::Permutation nb::to_odu_permutation(llvm::StringRef str) {
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

nb::ActivationType nb::to_activation_type(llvm::StringRef str) {
    if (!str.size() || isEqual(str, "None")) {
        return nb::ActivationType::None;
    }
    if (isEqual(str, "LeakyReLU")) {
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
    default:
        return "Unknown";
    }
}

std::string nb::to_string(CaseType case_) {
    switch (case_) {
    case CaseType::ZMajorConvolution:
        return "ZMajorConvolution";
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
    case CaseType::ActShave:
        return "ActShave";
    case CaseType::RaceConditionDMA:
        return "RaceConditionDMA";
    case CaseType::RaceConditionDPU:
        return "RaceConditionDPU";
    default:
        return "unknown";
    }
}

nb::CaseType nb::to_case(llvm::StringRef str) {
    if (isEqual(str, "ZMajorConvolution"))
        return CaseType::ZMajorConvolution;
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
    if (isEqual(str, "ActShave")) {
        return CaseType::ActShave;
    }
    if (isEqual(str, "RaceConditionDMA"))
        return CaseType::RaceConditionDMA;
    if (isEqual(str, "RaceConditionDPU"))
        return CaseType::RaceConditionDPU;
    return CaseType::Unknown;
};

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

nb::InputLayer nb::TestCaseJsonDescriptor::loadInputLayer(llvm::json::Object* jsonObj) {
    nb::InputLayer result;

    std::string layerType = "input";

    auto* input = jsonObj->getObject(layerType);
    if (!input) {
        // TODO: Add exception/error
        return result;
    }

    auto* shape = input->getArray("shape");
    if (!shape) {
        // TODO: add exception/error log
        return result;
    }

    for (size_t i = 0; i < shape->size(); i++) {
        result.shape[i] = (*shape)[i].getAsInteger().getValue();
    }

    result.qp = loadQuantizationParams(input);
    result.dtype = to_dtype(input->getString("dtype").getValue().str());

    return result;
}

nb::WeightLayer nb::TestCaseJsonDescriptor::loadWeightLayer(llvm::json::Object* jsonObj) {
    nb::WeightLayer result;

    std::string layerType = "weight";

    auto* weight = jsonObj->getObject(layerType);
    if (!weight) {
        // TODO: Add exception/error
        return result;
    }

    auto* shape = weight->getArray("shape");
    if (!shape) {
        // TODO: add exception/error log
        return result;
    }

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

nb::OutputLayer nb::TestCaseJsonDescriptor::loadOutputLayer(llvm::json::Object* jsonObj) {
    nb::OutputLayer result;

    auto* output = jsonObj->getObject("output");
    if (!output) {
        // TODO: add exception/error log
        return result;
    }

    auto* shape = output->getArray("shape");
    if (!shape) {
        // TODO: Add exception/error
        return result;
    }
    for (size_t i = 0; i < shape->size(); i++) {
        result.shape[i] = (*shape)[i].getAsInteger().getValue();
    }

    result.qp = loadQuantizationParams(output);
    result.dtype = to_dtype(output->getString("dtype").getValue().str());

    return result;
}

nb::ConvLayer nb::TestCaseJsonDescriptor::loadConvLayer(llvm::json::Object* jsonObj) {
    std::string layerType = "conv_op";

    nb::ConvLayer result;

    auto* op = jsonObj->getObject("conv_op");
    if (!op) {
        // TODO: add exception/error log
        return result;
    }

    auto* strides = op->getArray("stride");
    if (!strides) {
        // TODO: add exception/error log
        return result;
    }
    for (size_t i = 0; i < strides->size(); i++) {
        auto stride = (*strides)[i].getAsInteger();
        if (stride.hasValue()) {
            result.stride.at(i) = stride.getValue();
        }
    }

    auto* pads = op->getArray("pad");
    if (!pads) {
        // TODO: add exception/error log
        return result;
    }
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
    if (!op) {
        // TODO: add exception/error log
        return result;
    }

    auto* kernel_shape = op->getArray("kernel_shape");
    if (!kernel_shape) {
        // TODO: add exception/error log
        return result;
    }
    for (size_t i = 0; i < kernel_shape->size(); i++) {
        auto kernelsize = (*kernel_shape)[i].getAsInteger();
        if (kernelsize.hasValue()) {
            result.kernel_shape.at(i) = kernelsize.getValue();
        }
    }
    auto* strides = op->getArray("stride");
    if (!strides) {
        // TODO: add exception/error log
        return result;
    }
    for (size_t i = 0; i < strides->size(); i++) {
        auto stride = (*strides)[i].getAsInteger();
        if (stride.hasValue()) {
            result.stride.at(i) = stride.getValue();
        }
    }

    auto* pads = op->getArray("pad");
    if (!pads) {
        // TODO: add exception/error log
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

nb::TestCaseJsonDescriptor::TestCaseJsonDescriptor(llvm::StringRef jsonString) {
    if (!jsonString.empty()) {
        parse(jsonString);
    }
}

void nb::TestCaseJsonDescriptor::parse(llvm::StringRef jsonString) {
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
        throw std::runtime_error{llvm::formatv("HWTEST JSON parsing failed: {0}", err).str()};
    }

    auto* json_obj = exp->getAsObject();
    if (!json_obj) {
        throw std::runtime_error{"Expected to get JSON as an object"};
    }

    auto case_type = json_obj->getString("case_type");
    if (!case_type) {
        throw std::runtime_error{"Failed to get case type"};
    }

    caseType_ = nb::to_case(case_type.getValue());
    caseTypeStr_ = case_type.getValue().str();
    inLayer_ = loadInputLayer(json_obj);
    outLayer_ = loadOutputLayer(json_obj);

    // Load conv json attribute values. Similar implementation for ALL HW layers (DW, group conv, Av/Max pooling and
    // eltwise needed).
    if (caseType_ == CaseType::ZMajorConvolution || caseType_ == CaseType::DepthWiseConv ||
        caseType_ == CaseType::RaceConditionDPU) {
        wtLayer_ = loadWeightLayer(json_obj);
        convLayer_ = loadConvLayer(json_obj);

        if (caseType_ == CaseType::ZMajorConvolution) {
            odu_permutation_ = to_odu_permutation(json_obj->getString("output_order").getValue());
        }

        if (caseType_ == CaseType::RaceConditionDPU) {
            iterationCount_ = loadIterationCount(json_obj);
        }
        return;
    }

    if (caseType_ == CaseType::EltwiseAdd || caseType_ == CaseType::EltwiseMult) {
        wtLayer_ = loadWeightLayer(json_obj);
        return;
    }

    if (caseType_ == CaseType::MaxPool || caseType_ == CaseType::AvgPool) {
        poolLayer_ = loadPoolLayer(json_obj);
        return;
    }

    if (caseType_ == CaseType::ActShave) {
        activationLayer_ = loadActivationLayer(json_obj);
        return;
    }

    if (caseType_ == CaseType::RaceConditionDMA) {
        iterationCount_ = loadIterationCount(json_obj);
        return;
    }

    throw std::runtime_error{llvm::formatv("Unsupported case type: {0} 1", caseTypeStr_).str()};
}

nb::CaseType nb::TestCaseJsonDescriptor::loadCaseType(llvm::json::Object* jsonObj) {
    return to_case(jsonObj->getString("case_type").getValue().str());
}
