//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
    SparseZMajorConvolution,
    DepthWiseConv,
    EltwiseAdd,
    EltwiseMult,
    MaxPool,
    AvgPool,
    DifferentClustersDPU,
    MultiClustersDPU,
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
    M2iTask,
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

enum class SegmentationType { SOK = 1, SOH = 2 };

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
    std::vector<std::size_t> outputClusters;
    std::size_t weightsCluster;
    std::size_t weightsTableCluster;
};

struct MultiClusterDPUParams {
    std::vector<std::size_t> taskClusters;
    SegmentationType segmentation;
    bool broadcast;
};

struct OutputLayer {
    std::array<std::int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

enum class ActivationType {
    None,
    ReLU,
    ReLUX,
    LeakyReLU,
    Mish,
    HSwish,
    Sigmoid,
    Softmax,
    vau_sigm,
    vau_sqrt,
    vau_tanh,
    vau_log,
    vau_exp,
    lsu_b16,
    lsu_b16_vec,
    vau_dp4,
    vau_dp4a,
    vau_dp4m,
    sau_dp4,
    sau_dp4a,
    sau_dp4m,

    Unknown
};

ActivationType to_activation_type(llvm::StringRef str);
std::string to_string(ActivationType activationType);

struct ActivationLayer {
    ActivationType activationType = ActivationType::None;
    double alpha = 0.;
    double maximum = 0;
    size_t axis = 0;
    // TODO: add support for activation functions that take parameters
};

// M2I i/o color formats
enum class M2iFmt {
    PL_YUV420_8,
    SP_NV12_8,
    PL_FP16_RGB,
    PL_FP16_BGR,
    PL_RGB24,
    PL_BGR24,
    IL_RGB888,
    IL_BGR888,
    Unknown
    // Note: other less common formats exist
};
M2iFmt to_m2i_fmt(llvm::StringRef str);

struct M2iLayer {
    bool doCsc;   // do color-space-conversion
    bool doNorm;  // do normalization
    M2iFmt iFmt;
    M2iFmt oFmt;
    std::vector<int> outSizes;     // output sizes (optional)
    std::vector<float> normCoefs;  // normalization coefs (optional)
};

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    TestCaseJsonDescriptor(llvm::json::Object jsonObject);
    void parse(llvm::json::Object jsonObject);
    llvm::SmallVector<InputLayer> getInputLayerList() const {
        return inLayers_;
    }
    WeightLayer getWeightLayer() const {
        return wtLayer_;
    }
    llvm::SmallVector<OutputLayer> getOutputLayers() const {
        return outLayers_;
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
    M2iLayer getM2iLayer() const {
        return m2iLayer_;
    }
    RaceConditionParams getRaceConditionParams() const {
        return raceConditionParams_;
    }
    DPUTaskParams getDPUTaskParams() const {
        return DPUTaskParams_;
    }
    MultiClusterDPUParams getMultiClusterDPUParams() const {
        return multiClusterDPUParams_;
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

    vpux::VPU::ArchKind getArchitecture() const {
        return architecture_;
    }

private:
    llvm::SmallVector<InputLayer> loadInputLayer(llvm::json::Object* jsonObj);
    WeightLayer loadWeightLayer(llvm::json::Object* jsonObj);
    llvm::SmallVector<OutputLayer> loadOutputLayer(llvm::json::Object* jsonObj);
    DMAparams loadDMAParams(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    PoolLayer loadPoolLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    M2iLayer loadM2iLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);
    QuantParams loadQuantizationParams(llvm::json::Object* obj);
    RaceConditionParams loadRaceConditionParams(llvm::json::Object* obj);
    DPUTaskParams loadDPUTaskParams(llvm::json::Object* obj);
    MultiClusterDPUParams loadMultiClusterDPUParams(llvm::json::Object* obj);
    std::size_t loadIterationCount(llvm::json::Object* obj);
    std::size_t loadClusterNumber(llvm::json::Object* obj);

    CaseType caseType_;
    DMAparams DMAparams_;
    ConvLayer convLayer_;
    PoolLayer poolLayer_;
    llvm::SmallVector<InputLayer> inLayers_;
    WeightLayer wtLayer_;
    llvm::SmallVector<OutputLayer> outLayers_;
    ActivationLayer activationLayer_;
    M2iLayer m2iLayer_;
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
    MultiClusterDPUParams multiClusterDPUParams_;
    vpux::VPU::ArchKind architecture_;
};

}  // namespace nb
