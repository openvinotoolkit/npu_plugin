//
// Copyright Intel Corporation.
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

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/init.hpp"

#include <file_utils.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

namespace {

constexpr int64_t numDPU = 5;
constexpr int64_t maxSplitNum = 5;
constexpr int64_t initDimensionValue = 16;
constexpr int64_t maxDimensionValue = 64;
constexpr int64_t testStep = 2;

struct NceOpTensorShape {
    NceOpTensorShape(vpux::ShapeRef input, vpux::ShapeRef output): inputShape(input.raw()), outputShape(output.raw()) {
    }
    vpux::Shape inputShape;
    vpux::Shape outputShape;
};

vpux::SmallString getVPUNNModelFile() {
    vpux::SmallString modelDir;
    // probe for OpenVINO runtime dir
    auto ovBuildDir = InferenceEngine::getIELibraryPath();
    modelDir = ovBuildDir;
    llvm::sys::path::append(modelDir, "vpunn");
    llvm::sys::path::append(modelDir, "vpu_2_0.vpunn");
    VPUX_THROW_UNLESS(llvm::sys::fs::exists(modelDir), "vpunn model {0} does not exist", modelDir);
    return modelDir;
}

vpux::VPUIP::WorkloadCostParams buildWorkloadCost(const NceOpTensorShape& tensorShape, vpux::VPU::MPEMode mpeMode,
                                                  mlir::MLIRContext* ctx) {
    const auto outputShape = tensorShape.outputShape;
    const auto inputShape = tensorShape.inputShape;
    vpux::VPUIP::WorkloadCostParams costParams;
    costParams.dataType = mlir::Float16Type::get(ctx);
    costParams.inputShape = tensorShape.inputShape;
    costParams.outputShape = tensorShape.outputShape;
    costParams.padInfo = vpux::PadInfo(0, 0, 0, 0);
    costParams.kernelSize = {1, 1};
    costParams.kernelStride = {1, 1};
    costParams.nceTaskType = vpux::VPUIP::NCETaskType::CONV;
    costParams.arch = vpux::VPU::ArchKind::KMB;
    costParams.mpeMode = mpeMode;
    costParams.numDPU = numDPU;
    return costParams;
}
}  // namespace

TEST(MLIR_VPU_WorkloadCost, VPUNNCostInterface) {
    mlir::MLIRContext ctx;

    llvm::SmallVector<vpux::VPU::MPEMode> mpeModeList{vpux::VPU::MPEMode::VECTOR_FP16, vpux::VPU::MPEMode::VECTOR,
                                                      vpux::VPU::MPEMode::MATRIX, vpux::VPU::MPEMode::CUBOID_4x16};

    auto modelPath = getVPUNNModelFile();
    VPUNN::VPUCostModel costModel(modelPath.str().str());
    llvm::SmallVector<NceOpTensorShape> testTensorLists;
    for (int64_t h = initDimensionValue; h < maxDimensionValue; h *= testStep) {
        for (int64_t w = initDimensionValue; w < maxDimensionValue; w *= testStep) {
            for (int64_t c = initDimensionValue; c < maxDimensionValue; c *= testStep) {
                NceOpTensorShape tensorShape(vpux::ShapeRef({1, c, h, w}), vpux::ShapeRef({1, c, h, w}));
                testTensorLists.push_back(std::move(tensorShape));
            }
        }
    }
    for (auto& testTensor : testTensorLists) {
        for (auto& mpeMode : mpeModeList) {
            auto costParams = buildWorkloadCost(testTensor, mpeMode, &ctx);
            vpux::VPUIP::DpuTiler dpuTiler(costParams.outputShape, mpeMode, costModel);
            const auto& splitPool = dpuTiler.generateSplitNumberPool(numDPU, maxSplitNum);

            vpux::Shape nTilesOnDim(costParams.outputShape.size(), 1);
            const auto& outTilesWithSingleSplit = vpux::fillDividedTiles(nTilesOnDim, costParams.outputShape);
            auto baseHardwareExecutionCost = dpuTiler.cost(outTilesWithSingleSplit, costParams);

            dpuTiler.tileOverH(numDPU);
            for (auto& splitNum : splitPool) {
                dpuTiler.tileOverZ(splitNum);
            }

            const auto& splitCandidates = dpuTiler.getSplitPool();
            for (size_t idx = 0; idx < splitCandidates.size(); idx++) {
                auto hardwareExecutionCost = dpuTiler.cost(splitCandidates[idx], costParams);
                EXPECT_LE(hardwareExecutionCost, baseHardwareExecutionCost);
            }
        }
    }
}
