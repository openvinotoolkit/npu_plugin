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
    if (isEqual(str, "int4"))
        return nb::DType::I4;
    if (isEqual(str, "int8"))
        return nb::DType::I8;
    if (isEqual(str, "fp8"))
        return nb::DType::FP8;
    if (isEqual(str, "fp16"))
        return nb::DType::FP16;
    if (isEqual(str, "fp32"))
        return nb::DType::FP32;
    if (isEqual(str, "bf16"))
        return nb::DType::BF16;

    return nb::DType::UNK;
}

std::string nb::to_string(nb::DType dtype) {
    switch (dtype) {
    case nb::DType::U8:
        return "uint8";
    case nb::DType::I4:
        return "int4";
    case nb::DType::I8:
        return "int8";
    case nb::DType::FP8:
        return "fp8";
    case nb::DType::FP16:
        return "fp16";
    case nb::DType::FP32:
        return "fp32";
    case nb::DType::BF16:
        return "bf16";
    default:
        return "UNK";
    }
}

std::string nb::to_string(CaseType case_) {
    switch (case_) {
    case CaseType::conv2du8:
        return "conv2du8";
    case CaseType::conv2du8tofp16:
        return "conv2du8tofp16";
    case CaseType::conv2du8tobf8:
        return "conv2du8tobf8";
    case CaseType::conv2dfp16:
        return "conv2dfp16";
    case CaseType::conv2dfp16tobf16:
        return "conv2dfp16tobf16";
    case CaseType::conv2dfp16tou8:
        return "conv2dfp16tou8";
    case CaseType::conv2dbf16:
        return "conv2dbf16";
    case CaseType::conv2dbf16tofp16:
        return "conv2dbf16tofp16";
    case CaseType::conv2dbf16tou8:
        return "conv2dbf16tou8";
    case CaseType::conv2dbf16tobf8:
        return "conv2dbf16tobf8";
    case CaseType::conv2dbf8tobf8:
        return "conv2dbf8tobf8";
    case CaseType::conv2dbf8tobf16:
        return "conv2dbf8tobf16";
    case CaseType::conv2dbf8tofp16:
        return "conv2dbf8tofp16";
    case CaseType::conv2dbf8tou8:
        return "conv2dbf8tou8";
    case CaseType::elementwiseU8toU8:
        return "elementwiseU8toU8";
    case CaseType::elementwiseI8toI8:
        return "elementwiseI8toI8";
    case CaseType::avpoolfp16tofp16:
        return "avpoolfp16tofp16";
    case CaseType::avpoolI8toI8:
        return "avpoolI8toI8";
    case CaseType::maxpoolfp16tofp16:
        return "maxpoolfp16tofp16";
    case CaseType::maxpoolI8toI8:
        return "maxpoolI8toI8";
    default:
        return "unknown";
    }
}

nb::CaseType nb::to_case(llvm::StringRef str) {
    if (isEqual(str, "conv2du8"))
        return CaseType::conv2du8;
    if (isEqual(str, "conv2du8tofp16"))
        return CaseType::conv2du8tofp16;
    if (isEqual(str, "conv2du8tobf8"))
        return CaseType::conv2du8tobf8;
    if (isEqual(str, "conv2dfp16"))
        return CaseType::conv2dfp16;
    if (isEqual(str, "conv2dfp16tobf16"))
        return CaseType::conv2dfp16tobf16;
    if (isEqual(str, "conv2dfp16tou8"))
        return CaseType::conv2dfp16tou8;
    if (isEqual(str, "conv2dbf16"))
        return CaseType::conv2dbf16;
    if (isEqual(str, "conv2dbf16tofp16"))
        return CaseType::conv2dbf16tofp16;
    if (isEqual(str, "conv2dbf16tou8"))
        return CaseType::conv2dbf16tou8;
    if (isEqual(str, "conv2dbf16tobf8"))
        return CaseType::conv2dbf16tobf8;
    if (isEqual(str, "conv2dbf8tobf8"))
        return CaseType::conv2dbf8tobf8;
    if (isEqual(str, "conv2dbf8tobf16"))
        return CaseType::conv2dbf8tobf16;
    if (isEqual(str, "conv2dbf8tofp16"))
        return CaseType::conv2dbf8tofp16;
    if (isEqual(str, "conv2dbf8tou8"))
        return CaseType::conv2dbf8tou8;
    if (isEqual(str, "elementwiseU8toU8"))
        return CaseType::elementwiseU8toU8;
    if (isEqual(str, "elementwiseI8toI8"))
        return CaseType::elementwiseI8toI8;
    if (isEqual(str, "avpoolfp16tofp16"))
        return CaseType::avpoolfp16tofp16;
    if (isEqual(str, "apoolI8toI8"))
        return CaseType::avpoolI8toI8;
    if (isEqual(str, "maxpoolfp16tofp16"))
        return CaseType::maxpoolfp16tofp16;
    if (isEqual(str, "maxpoolI8toI8"))
        return CaseType::maxpoolI8toI8;
    return CaseType::Unknown;
};

nb::IWLayer nb::TestCaseJsonDescriptor::loadIWLayer(llvm::json::Object* jsonObj, std::string layerType) {
    nb::IWLayer result;

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

    result.qp.scale = input->getNumber("scale").getValue();
    result.qp.zeropoint = input->getInteger("zeropoint").getValue();

    auto* dg = input->getObject("data_generator");
    if (!dg) {
        // TODO: Add exception/error
        return result;
    }

    result.dg.name = dg->getString("name").getValue().str();
    result.dg.dtype = to_dtype(dg->getString("dtype").getValue().str());
    result.dg.low_range = dg->getInteger("low_range").getValue();
    result.dg.high_range = dg->getInteger("high_range").getValue();

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

    result.qp.scale = output->getNumber("scale").getValue();
    result.qp.zeropoint = output->getInteger("zeropoint").getValue();
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

    return result;
}

nb::ActivationLayer nb::TestCaseJsonDescriptor::loadActivationLayer(llvm::json::Object* jsonObj) {
    nb::ActivationLayer result;

    auto* act = jsonObj->getObject("activation");
    if (!act) {
        // TODO: add exception/error log
        return result;
    }

    result.activationType = act->getString("name").getValue().str();

    auto alpha = act->getNumber("alpha");
    if (alpha.hasValue()) {
        result.alpha = alpha.getValue();
    }

    return result;
}

nb::TestCaseJsonDescriptor::TestCaseJsonDescriptor(llvm::StringRef jsonString): caseType_(CaseType::Unknown) {
    if (!jsonString.empty()) {
        if (!parse(jsonString)) {
            throw std::exception();
        }
    }
}

bool nb::TestCaseJsonDescriptor::parse(llvm::StringRef jsonString) {
    if (jsonString.empty()) {
        return false;
    }

    llvm::Expected<llvm::json::Value> exp = llvm::json::parse(jsonString);
    if (!exp) {
        return false;
    }

    auto* json_obj = exp->getAsObject();
    if (!json_obj) {
        // TODO: add error log
        return false;
    }

    auto case_type = json_obj->getString("case_type");
    if (!case_type) {
        return false;
    }

    caseType_ = nb::to_case(case_type.getValue());

    // Load conv json attribute values. Similar implementation for ALL HW layers (DW, group conv, Av/Max pooling and
    // eltwise needed).
    if (case_type.getValue().str().find("Conv") != std::string::npos) {
        inLayer_ = loadIWLayer(json_obj, "input");
        wtLayer_ = loadIWLayer(json_obj, "weight");
        outLayer_ = loadOutputLayer(json_obj);
        convLayer_ = loadConvLayer(json_obj);

        auto* activation = json_obj->getObject("activation");
        if (activation && activation->getString("name").hasValue()) {
            hasActivationLayer_ = true;
            activationLayer_ = loadActivationLayer(json_obj);
        } else {
            hasActivationLayer_ = false;
        }
        return true;
    }

    return false;
}

nb::CaseType nb::TestCaseJsonDescriptor::loadCaseType(llvm::json::Object* jsonObj) {
    return to_case(jsonObj->getString("case_type").getValue().str());
}
