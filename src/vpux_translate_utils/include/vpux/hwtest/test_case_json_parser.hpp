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

#include <array>
#include <fstream>
#include <set>
#include <string>

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
enum class CaseType {
    conv2du8,
    conv2du8tofp16,
    conv2du8tobf8,
    conv2dfp16,
    conv2dfp16tobf16,
    conv2dfp16tou8,
    conv2dbf16,
    conv2dbf16tofp16,
    conv2dbf16tou8,
    conv2dbf16tobf8,
    conv2dbf8tobf8,
    conv2dbf8tobf16,
    conv2dbf8tofp16,
    conv2dbf8tou8,
    elementwiseU8toU8,
    elementwiseI8toI8,
    avpoolfp16tofp16,
    avpoolI8toI8,
    maxpoolfp16tofp16,
    maxpoolI8toI8,
    Unknown
};

std::string to_string(CaseType case_);
CaseType to_case(llvm::StringRef str);

enum class DType { I4, U8, I8, FP8, FP16, FP32, BF16, UNK };

DType to_dtype(llvm::StringRef str);
std::string to_string(DType dtype);

struct QuantParams {
    double scale = 0.;
    int64_t zeropoint = 0;
};

struct DataGenerator {
    std::string name;
    DType dtype = DType::UNK;
    int64_t low_range = 0;
    int64_t high_range = 0;
};

struct Shape {};

// Input and weight layers have similar structure in rtl config descriptor
struct IWLayer {
    DataGenerator dg;
    QuantParams qp;
    std::array<int64_t, 4> shape = {0};
};

struct ConvLayer {
    std::array<int64_t, 2> stride = {0};
    std::array<int64_t, 4> pad = {0};
    int64_t group = 0;
    int64_t dilation = 0;
};

struct OutputLayer {
    std::array<int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

struct ActivationLayer {
    std::string activationType;
    double alpha = 0.;
    // TODO: add support for activation functions that take parameters
};

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    bool parse(llvm::StringRef jsonString);
    IWLayer getInputLayer() const {
        return inLayer_;
    }
    IWLayer getWeightLayer() const {
        return wtLayer_;
    }
    OutputLayer getOutputLayer() const {
        return outLayer_;
    }
    ConvLayer getConvLayer() const {
        return convLayer_;
    }
    ActivationLayer getActivationLayer() const {
        return activationLayer_;
    }
    CaseType getCaseType() const {
        return caseType_;
    }

private:
    IWLayer loadIWLayer(llvm::json::Object* jsonObj, std::string layerType);
    OutputLayer loadOutputLayer(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);

    CaseType caseType_;
    ConvLayer convLayer_;
    IWLayer inLayer_;
    IWLayer wtLayer_;
    OutputLayer outLayer_;
    ActivationLayer activationLayer_;
    bool hasActivationLayer_;
};

}  // namespace nb
