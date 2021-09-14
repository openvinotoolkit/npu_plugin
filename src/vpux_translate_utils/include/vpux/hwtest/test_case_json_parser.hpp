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

#pragma once

#include <array>
#include <fstream>
#include <set>
#include <string>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)  // integer conversion mismatch
#pragma warning(disable : 4267)  // size_t to integer conversion
#pragma warning(disable : 4624)  // destructor implicitly defined as deleted
#endif

#include "llvm/Support/JSON.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace nb {
enum class CaseType { ZMajorConvolution, EltwiseAdd, EltwiseMult, MaxPool, Unknown };

std::string to_string(CaseType case_);
CaseType to_case(llvm::StringRef str);

enum class DType { U4, I4, U8, I8, FP8, FP16, FP32, BF16, UNK };

DType to_dtype(llvm::StringRef str);
std::string to_string(DType dtype);

struct QuantParams {
    bool present = false;
    double scale = 0.;
    std::int64_t zeropoint = 0;
    std::int64_t low_range = 0;
    std::int64_t high_range = 0;
};

struct Shape {};

// Input and weight layers have similar structure in rtl config descriptor
struct InputLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    std::array<std::int64_t, 4> shape = {0};
};

struct WeightLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    std::array<std::int64_t, 4> shape = {0};
    std::string filename;
};

struct ConvLayer {
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
};

struct PoolLayer {
    std::string pool_type = "max";
    std::array<std::int64_t, 2> kernel_shape = {0};
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
};

struct OutputLayer {
    std::array<std::int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

enum class ActivationType { None, ReLU, ReLUX, LeakyReLU, Mish, Unknown };

ActivationType to_activation_type(llvm::StringRef str);
std::string to_string(ActivationType activationType);

struct ActivationLayer {
    ActivationType activationType = ActivationType::None;
    double alpha = 0.;
    double maximum = 0;
    // TODO: add support for activation functions that take parameters
};

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    void parse(llvm::StringRef jsonString);
    InputLayer getInputLayer() const {
        return inLayer_;
    }
    WeightLayer getWeightLayer() const {
        return wtLayer_;
    }
    OutputLayer getOutputLayer() const {
        return outLayer_;
    }
    ConvLayer getConvLayer() const {
        return convLayer_;
    }
    PoolLayer getPoolLayer() const {
        return poolLayer_;
    }
    ActivationLayer getActivationLayer() const {
        return activationLayer_;
    }
    CaseType getCaseType() const {
        return caseType_;
    }
    llvm::StringRef getKernelFilename() const {
        return kernelFilename_;
    }
    std::string getCaseStr() const {
        return caseTypeStr_;
    };
    vpux::VPUIP::PPELayerType getPPELayerType() const {
        return ppeLayerType_;
    }

private:
    InputLayer loadInputLayer(llvm::json::Object* jsonObj);
    WeightLayer loadWeightLayer(llvm::json::Object* jsonObj);
    OutputLayer loadOutputLayer(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    PoolLayer loadPoolLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);
    QuantParams loadQuantizationParams(llvm::json::Object* obj);

    CaseType caseType_;
    ConvLayer convLayer_;
    PoolLayer poolLayer_;
    InputLayer inLayer_;
    WeightLayer wtLayer_;
    OutputLayer outLayer_;
    ActivationLayer activationLayer_;
    bool hasActivationLayer_;
    std::string kernelFilename_;
    std::string caseTypeStr_;
    vpux::VPUIP::PPELayerType ppeLayerType_ = vpux::VPUIP::PPELayerType::ADD;
};

}  // namespace nb
