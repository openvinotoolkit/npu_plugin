//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

//                         comp       decomp
//       [CMXbuf0Uncomp] --------   --------- [CMXbuf1Uncomp]
//              |               |   |                |
//              | 1:1           |   |                | 1:1
//              |               |   |                |
//      [DDRinput_uncomp]  [DDRspilledComp]  [DDRoutput_uncomp]
//

void buildDMACompressAct(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                         Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    // set runtime resources
    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, 1, None, None,
                                           log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    auto input = testDesc.getInputLayerList().front();
    auto dmaParams = testDesc.getDMAparams();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    // Activation compression enabled - Dense Mode
    // Compiler must ensure that the DDR allocation is capable of handling the worst case compressed size (which can be
    // more than the source)
    // DTS = X * Y * Z * (element size in bytes)
    // denseSize = (DTS * (65/64)) + 1
    // DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    const auto alignment = Byte(32);
    const auto elementSizeBytes =
            getElemTypeSize(inputType).count() < CHAR_BIT ? 1 : getElemTypeSize(inputType).count() / CHAR_BIT;
    const auto denseTensorSize = inShape[vpux::Dims4D::Act::C.ind()] * inShape[vpux::Dims4D::Act::W.ind()] *
                                 inShape[vpux::Dims4D::Act::H.ind()] * elementSizeBytes;
    const auto denseSize = static_cast<uint64_t>(denseTensorSize * (static_cast<float>(65) / 64) + 1);
    const auto DDRspilledCompShape = vpux::alignValUp(denseSize, static_cast<std::uint64_t>(alignment.count()));

    VPUX_THROW_UNLESS(!inShape.empty(), "buildDMACompressAct: Input rank is 0");
    VPUX_THROW_UNLESS(inShape == outShape, "buildDMACompressAct: in_shape and out_shape don't match");
    VPUX_THROW_UNLESS(inputType == outputType, "buildDMACompressAct: outputType and outputType don't match");

    auto inputTotalSize = totalTensorSize(inShape, inputType);

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    const auto DDRspilledCompType =
            getMemRefType(VPURT::BufferSection::NetworkOutput, DDRspilledCompShape, inputType, DimsOrder::C);

    const auto funcType =
            builder.getFunctionType(makeArrayRef(std::vector<mlir::Type>{inType, DDRspilledCompType, outType}),
                                    makeArrayRef(std::vector<mlir::Type>{DDRspilledCompType, outType}));

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), printToString("dma_compress_activations"),
                                                   funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    size_t CMX0_AVAILABLE_OFFSET = 0;
    int barrierNumber = 0;

    // DDRinput_uncomp - CMXbuf0Uncomp
    auto DDRinput_uncomp = func.getArgument(0);

    const auto sectionIdx = 0;
    auto CMXbuf0UncompType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, inShape, inputType, DimsOrder::NHWC);
    auto CMXbuf0Uncomp = createDeclareTensorOp(funcbuilder, CMXbuf0UncompType, VPURT::BufferSection::CMX_NN, sectionIdx,
                                               CMX0_AVAILABLE_OFFSET);
    CMX0_AVAILABLE_OFFSET += inputTotalSize;

    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), DDRinput_uncomp,
                                          CMXbuf0Uncomp.getOperation()->getResult(0), dmaParams.engine);

    // act_compression_entry
    enum { actCompressionEntrySize = 32 };

    // E#74119
    // RT parsing stage doesn't know the real address for actCompressionEntry, so RT will hardcode the address
    // Temporary workaround: actCompressionEntry allocation is done at the end of the CMX
    auto ops = module.getOps<IE::MemoryResourceOp>();

    auto cmxMemResourceOp = llvm::find_if(ops, [](IE::MemoryResourceOp it) {
        return it.sym_name().equals("CMX_NN");
    });
    VPUX_THROW_UNLESS(cmxMemResourceOp != ops.end(), "buildDMACompressAct: CMX_NN not found as MemoryResource");

    const auto cmxWorkspaceSize = (*cmxMemResourceOp).byteSize();
    const auto newCmxWorkspaceSize = cmxWorkspaceSize - actCompressionEntrySize;
    const auto newCmxWorkspaceSizeAttr = getIntAttr(ctx, newCmxWorkspaceSize);
    (*cmxMemResourceOp).byteSizeAttr(newCmxWorkspaceSizeAttr);

    const auto elemType = getUInt8Type(ctx);
    auto actCompressionEntryType = getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx,
                                                 ShapeRef({actCompressionEntrySize}), elemType, DimsOrder::C);
    auto actCompressionEntry = createDeclareTensorOp(funcbuilder, actCompressionEntryType, VPURT::BufferSection::CMX_NN,
                                                     sectionIdx, newCmxWorkspaceSize);

    // CMXbuf0Uncomp - DDRspilledComp
    auto DDRspilledComp = func.getArgument(1);

    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::CompressDMAOp>(funcbuilder, mlir::ValueRange(barrier0.barrier()),
                                                mlir::ValueRange(barrier1.barrier()), builder.getUnknownLoc(),
                                                CMXbuf0Uncomp.getOperation()->getResult(0),
                                                actCompressionEntry.getOperation()->getResult(0), DDRspilledComp,
                                                dmaParams.engine, nullptr, false, false, nullptr);

    // DDRspilledComp - CMXbuf1Uncomp
    auto CMXbuf1UncompType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, inShape, inputType, DimsOrder::NHWC);
    auto CMXbuf1Uncomp = createDeclareTensorOp(funcbuilder, CMXbuf1UncompType, VPURT::BufferSection::CMX_NN, sectionIdx,
                                               CMX0_AVAILABLE_OFFSET);
    CMX0_AVAILABLE_OFFSET += inputTotalSize;

    auto barrier2 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::DecompressDMAOp>(
            funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(barrier2.barrier()),
            builder.getUnknownLoc(), DDRspilledComp, actCompressionEntry.getOperation()->getResult(0),
            CMXbuf1Uncomp.getOperation()->getResult(0), dmaParams.engine, nullptr, false, false, nullptr);

    // CMXbuf1Uncomp - DDRoutput_uncomp
    auto DDRoutput_uncomp = func.getArgument(2);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier2.barrier()), mlir::ValueRange(),
                                          builder.getUnknownLoc(), CMXbuf1Uncomp.getOperation()->getResult(0),
                                          DDRoutput_uncomp, dmaParams.engine);

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                             mlir::ValueRange{DDRspilledComp, DDRoutput_uncomp});

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(DDRspilledCompShape), inputType, DimsOrder::C, nullptr),
                getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
