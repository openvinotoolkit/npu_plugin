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

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}

TEST(MLIR_ShapedPropertiesTypeInterface, DistributedBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        !InputDistributed = type !VPUIP.DistributedBuffer<
            1x32x16x16xf16, #NHWC, @CMX_NN, {
            mode = "overlapped",
            num_tiles = [1, 1, 4, 1],
            kernel = [3, 3],
            pads = {bottom = 1, left = 1, right = 1, top = 1}
        }>

        !Input_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>
        !Output_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>

        !InputStub_CMX = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>

        module @main {
            func @main(%input: !Input_DDR) -> !Output_DDR {
                %input_cmx = VPURT.AllocDistributed -> !InputDistributed
                %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR)
                                            outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
                    %1 = IERT.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
                }

                %output = memref.alloc() : !Output_DDR
                %2 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg0: !InputStub_CMX)
                                            outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
                    %3 = IERT.Copy { out_mem_space = @DDR } inputs(%arg0: !InputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
                }

                return %output: !Output_DDR
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    for (auto& op : func.getOps()) {
        auto clusterTiling = mlir::dyn_cast<vpux::VPUIP::NCEClusterTilingOp>(op);
        if(clusterTiling == nullptr) {
            continue;
        }

        for (auto operand: clusterTiling.getOperands()) {
            auto distributedBuffer = operand.getType().dyn_cast<vpux::VPUIP::DistributedBufferType>();
            if(distributedBuffer == nullptr) {
                continue;
            }

            auto shapedType = distributedBuffer.dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedType != nullptr) << "Operand is DistributedBufferType but not supports vpux::ShapedPropertiesTypeInterface";

            EXPECT_EQ(shapedType.getShape(), vpux::ShapeRef({1, 32, 16, 16}));
            EXPECT_EQ(shapedType.getMemShape(), vpux::MemShape({1, 16, 16, 32}));

            EXPECT_TRUE(shapedType.hasRank());
            EXPECT_EQ(shapedType.getRank(), 4);
            EXPECT_EQ(shapedType.getNumElements(), 32*16*16);

            EXPECT_TRUE(shapedType.getElementType().isa<mlir::Float16Type>());

            EXPECT_EQ(shapedType.getDimsOrder(), vpux::DimsOrder::NHWC);

            EXPECT_EQ(shapedType.getMemSpace().getLeafName(), CMX_NAME);
            EXPECT_EQ(shapedType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

            const SmallVector<vpux::Bit> strides({131072_Bit, 16_Bit, 8192_Bit, 512_Bit});
            const SmallVector<vpux::Bit> memStrides({131072_Bit, 8192_Bit, 512_Bit, 16_Bit});
            EXPECT_EQ(shapedType.getStrides().raw(), strides);
            EXPECT_EQ(shapedType.getMemStrides().raw(), memStrides);

            EXPECT_EQ(shapedType.getElemTypeSize().count(), 16);
            EXPECT_EQ(shapedType.getTotalAllocSize().count(), 16384);
            EXPECT_EQ(shapedType.getCompactAllocSize().count(), 16384);
        }
    }
}
