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
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

namespace {

constexpr int64_t numDPU = 5;
constexpr int64_t maxSplitNum = 5;

vpux::VPUIP::WorkloadCostParams buildWorkloadCost(vpux::VPU::NCEConvolutionOp convolutionOp,
                                                  vpux::VPU::MPEMode mpeMode) {
    const auto outputShape = vpux::getShape(convolutionOp.output());
    const auto inputShape = vpux::getShape(convolutionOp.input());
    const auto kernelStrides = vpux::parseIntArrayAttr<int64_t>(convolutionOp.strides());
    const auto filterShape = convolutionOp.rawFilterShapeAttr() != nullptr
                                     ? vpux::Shape(vpux::parseIntArrayAttr<int64_t>(convolutionOp.rawFilterShapeAttr()))
                                     : vpux::getShape(convolutionOp.filter());

    vpux::VPUIP::WorkloadCostParams costParams;
    costParams.dataType = convolutionOp.input().getType().template cast<mlir::RankedTensorType>().getElementType();
    costParams.inputShape = inputShape.raw();
    costParams.outputShape = outputShape.raw();
    costParams.padInfo = vpux::VPU::toPadInfo(convolutionOp.pad());
    costParams.kernelSize = {filterShape[vpux::Dims4D::Filter::KY], filterShape[vpux::Dims4D::Filter::KX]};
    costParams.kernelStride = {kernelStrides[0], kernelStrides[1]};
    costParams.nceTaskType = vpux::VPUIP::NCETaskType::CONV;
    costParams.arch = vpux::VPU::ArchKind::KMB;
    costParams.mpeMode = mpeMode;
    costParams.numDPU = numDPU;
    return costParams;
}

}  // namespace

TEST(MLIR_VPU_WorkloadCost, VPUNNCostInterface) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @main {
            func @main(%arg0: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> =
                    #const.Content<dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]>

                %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x64x64xf16, {order = #NHWC}>
                    -> tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %1 = IE.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<64x64x1x1xf16, {order = #NHWC}>
                    -> tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                %2 = VPU.NCE.Convolution(%0, %1) {
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        strides = [1, 1]
                    } : tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    -> tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

                %3 = IE.Copy(%2) : tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    -> tensor<1x64x64x64xf16, {order = #NHWC}>

                return %3 : tensor<1x64x64x64xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    llvm::SmallVector<vpux::VPU::MPEMode> mpeModeList{vpux::VPU::MPEMode::VECTOR_FP16, vpux::VPU::MPEMode::VECTOR,
                                                      vpux::VPU::MPEMode::MATRIX, vpux::VPU::MPEMode::CUBOID_4x16};

    for (auto& op : func.getOps()) {
        if (auto convolutionOp = mlir::dyn_cast<vpux::VPU::NCEConvolutionOp>(op)) {
            for (auto& mpeMode : mpeModeList) {
                auto costParams = buildWorkloadCost(convolutionOp, mpeMode);
                const auto outputShape = vpux::getShape(convolutionOp.output());
                vpux::VPUIP::DpuTiler dpuTiler(outputShape, mpeMode);
                const auto& splitPool = dpuTiler.generateSplitNumberPool(numDPU, maxSplitNum);

                vpux::Shape nTilesOnDim(outputShape.size(), 1);
                const auto& outTilesWithSingleSplit = vpux::fillDividedTiles(nTilesOnDim, outputShape);
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
}
