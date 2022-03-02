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

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/Support/JSON.h>

#include <array>
#include <fstream>
#include <set>
#include <string>

namespace nb {
enum class CaseType {
    DMA,
    ZMajorConvolution,
    DepthWiseConv,
    EltwiseAdd,
    EltwiseMult,
    MaxPool,
    AvgPool,
    DifferentClustersDPU,
    ActShave,
    ReadAfterWriteDPUDMA,
    ReadAfterWriteDMADPU,
    ReadAfterWriteACTDMA,
    ReadAfterWriteDMAACT,
    ReadAfterWriteDPUACT,
    ReadAfterWriteACTDPU,
    RaceConditionDMA,
    RaceConditionDPU,
    RaceConditionDPUDMA,
    RaceConditionDPUDMAACT,
    RaceCondition,
    MultiClusteringSOH,
    MultiClusteringSOK,
    Unknown
};

std::string to_string(CaseType case_);
CaseType to_case(llvm::StringRef str);

enum class DType { U4, I4, U8, I8, I32, FP8, FP16, FP32, BF16, UNK };

enum class MemoryLocation { CMX0, CMX1, DDR, Unknown };
MemoryLocation to_memory_location(llvm::StringRef str);
std::string to_string(MemoryLocation memoryLocation);

DType to_dtype(llvm::StringRef str);
std::string to_string(DType dtype);
MVCNN::Permutation to_odu_permutation(llvm::StringRef str);
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

struct DMAparams {
    MemoryLocation srcLocation;
    MemoryLocation dstLocation;
    int64_t engine;
};

struct ConvLayer {
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
    bool compress = false;
    vpux::VPU::MPEMode cube_mode = vpux::VPU::MPEMode::CUBOID_16x16;
};

struct PoolLayer {
    std::string pool_type = "max";
    std::array<std::int64_t, 2> kernel_shape = {0};
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
};

struct RaceConditionParams {
    size_t iterationsCount;
    size_t requestedClusters;

    // requested DPU's per cluster for DPUop case
    // requested ActShave's per cluster for ActShaveOp case
    // requested DMA engines for DMAop case
    size_t requestedUnits;
};

struct DPUTaskParams {
    std::size_t inputCluster;
    std::size_t outputCluster;
    std::size_t weightsCluster;
    std::size_t weightsTableCluster;
};

struct OutputLayer {
    std::array<std::int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

enum class ActivationType { None, ReLU, ReLUX, LeakyReLU, Mish, HSwish, Sigmoid, Softmax, Unknown };

ActivationType to_activation_type(llvm::StringRef str);
std::string to_string(ActivationType activationType);

struct ActivationLayer {
    ActivationType activationType = ActivationType::None;
    double alpha = 0.;
    double maximum = 0;
    size_t axis = 0;
    // TODO: add support for activation functions that take parameters
};

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    TestCaseJsonDescriptor(llvm::json::Object jsonObject);
    void parse(llvm::json::Object jsonObject);
    InputLayer getInputLayer() const {
        return inLayer_;
    }
    WeightLayer getWeightLayer() const {
        return wtLayer_;
    }
    OutputLayer getOutputLayer() const {
        return outLayer_;
    }
    DMAparams getDMAparams() const {
        return DMAparams_;
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
    RaceConditionParams getRaceConditionParams() const {
        return raceConditionParams_;
    }
    DPUTaskParams getDPUTaskParams() const {
        return DPUTaskParams_;
    }
    CaseType getCaseType() const {
        return caseType_;
    }
    llvm::StringRef getKernelFilename() const {
        return kernelFilename_;
    }
    std::string getCaseStr() const {
        return caseTypeStr_;
    }
    vpux::VPU::PPEMode getPPELayerType() const {
        return ppeLayerType_;
    }
    MVCNN::Permutation getODUPermutation() const {
        return odu_permutation_;
    }
    std::size_t getIterationCount() const {
        return iterationCount_;
    }
    std::size_t getClusterNumber() const {
        return clusterNumber_;
    }
    std::shared_ptr<TestCaseJsonDescriptor> getUnderlyingOp() const {
        return underlyingOp_;
    }

private:
    InputLayer loadInputLayer(llvm::json::Object* jsonObj);
    WeightLayer loadWeightLayer(llvm::json::Object* jsonObj);
    OutputLayer loadOutputLayer(llvm::json::Object* jsonObj);
    DMAparams loadDMAParams(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    PoolLayer loadPoolLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);
    QuantParams loadQuantizationParams(llvm::json::Object* obj);
    RaceConditionParams loadRaceConditionParams(llvm::json::Object* obj);
    DPUTaskParams loadDPUTaskParams(llvm::json::Object* obj);
    std::size_t loadIterationCount(llvm::json::Object* obj);
    std::size_t loadClusterNumber(llvm::json::Object* obj);

    CaseType caseType_;
    DMAparams DMAparams_;
    ConvLayer convLayer_;
    PoolLayer poolLayer_;
    InputLayer inLayer_;
    WeightLayer wtLayer_;
    OutputLayer outLayer_;
    ActivationLayer activationLayer_;
    bool hasActivationLayer_;
    std::string kernelFilename_;
    std::string caseTypeStr_;
    vpux::VPU::PPEMode ppeLayerType_ = vpux::VPU::PPEMode::ADD;
    MVCNN::Permutation odu_permutation_ = MVCNN::Permutation::Permutation_ZXY;
    std::size_t iterationCount_;
    std::size_t clusterNumber_;
    std::shared_ptr<TestCaseJsonDescriptor> underlyingOp_;
    RaceConditionParams raceConditionParams_;
    DPUTaskParams DPUTaskParams_;
};

}  // namespace nb
