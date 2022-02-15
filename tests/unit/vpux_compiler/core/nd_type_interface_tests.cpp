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

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

constexpr StringRef CMX_NAME = "CMX_NN";
constexpr StringRef DDR_NAME = "DDR";

namespace {

class TestUnrankedOp : public mlir::Op<TestUnrankedOp, mlir::OpTrait::ZeroRegion> {
public:
    using Op::Op;

    static constexpr llvm::StringLiteral getOperationName() {
        return llvm::StringLiteral("test.UnrankedOp");
    }

    static ArrayRef<llvm::StringRef> getAttributeNames() {
        return {};
    }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::TypeRange resultTypes,
                      mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes = {}) {
        state.addTypes(resultTypes);
        state.addOperands(operands);
        state.addAttributes(attributes);
    }
};

class TestDialect final : public mlir::Dialect {
public:
    explicit TestDialect(mlir::MLIRContext* ctx)
            : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<TestDialect>()) {
        addOperations<TestUnrankedOp>();
    }

    static constexpr llvm::StringLiteral getDialectNamespace() {
        return llvm::StringLiteral("test");
    }
};

}  // namespace

TEST(MLIR_NDTypeInterface, RankedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x16x16xf16> {
                %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]>
                %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x16x32x32xf16>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
                return %0 : tensor<1x16x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
            // Input type
            const auto ndInputType = convOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
            EXPECT_TRUE(ndInputType != nullptr) << "Input is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndInputType.getShape(), ShapeRef({1, 16, 32, 32}));
            EXPECT_EQ(ndInputType.getMemShape(), MemShape({1, 16, 32, 32}));

            EXPECT_TRUE(ndInputType.hasRank());
            EXPECT_EQ(ndInputType.getRank(), 4);
            EXPECT_EQ(ndInputType.getNumElements(), 16 * 32 * 32);

            EXPECT_TRUE(ndInputType.getElementType().isa<mlir::Float16Type>());

            EXPECT_EQ(ndInputType.getDimsOrder(), DimsOrder::NCHW);

            EXPECT_EQ(ndInputType.getMemSpace(), nullptr);
            EXPECT_EQ(ndInputType.getMemoryKind(), VPU::MemoryKind::DDR);

            const SmallVector<Bit> strides({262144_Bit, 16384_Bit, 512_Bit, 16_Bit});
            EXPECT_EQ(ndInputType.getStrides().raw(), strides);
            EXPECT_EQ(ndInputType.getMemStrides().raw(), strides);

            EXPECT_EQ(ndInputType.getElemTypeSize().count(), 16);
            EXPECT_EQ(ndInputType.getTotalAllocSize().count(), 32768);
            EXPECT_EQ(ndInputType.getCompactAllocSize().count(), 32768);

            // Filter type
            const auto ndFilterType = convOp.filter().getType().dyn_cast<vpux::NDTypeInterface>();
            EXPECT_TRUE(ndFilterType != nullptr) << "Filter is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndFilterType.getShape(), ShapeRef({16, 16, 1, 1}));
            EXPECT_EQ(ndFilterType.getMemShape(), MemShape({16, 1, 1, 16}));

            EXPECT_EQ(ndFilterType.getDimsOrder(), DimsOrder::NHWC);

            // Output type
            const auto ndOutputType = convOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
            EXPECT_TRUE(ndOutputType != nullptr) << "Output is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndOutputType.getShape(), ShapeRef({1, 16, 16, 16}));
            const SmallVector<int64_t> shape({1, 16, 64, 64});
            const auto newOutputShape = ndOutputType.changeShape(ShapeRef(shape));
            EXPECT_EQ(newOutputShape.getShape(), ShapeRef(shape));

            EXPECT_TRUE(ndOutputType.getElementType().isa<mlir::Float16Type>());
            const auto elemType = mlir::Float32Type::get(op.getContext());
            const auto newOutputElemType = ndOutputType.changeElemType(elemType);
            EXPECT_TRUE(newOutputElemType.getElementType().isa<mlir::Float32Type>());

            EXPECT_EQ(ndOutputType.getDimsOrder(), DimsOrder::NCHW);
            const auto dimsOrder = DimsOrder::NHWC;
            const auto newOutputDimsOrder = ndOutputType.changeDimsOrder(dimsOrder);
            EXPECT_EQ(newOutputDimsOrder.getDimsOrder(), dimsOrder);

            EXPECT_EQ(ndOutputType.getMemSpace(), nullptr);
            const auto memSpace = vpux::IndexedSymbolAttr::get(op.getContext(), CMX_NAME);
            auto newOutputMemSpace = ndOutputType.changeMemSpace(memSpace);
            EXPECT_EQ(newOutputMemSpace.getMemSpace().getLeafName(), CMX_NAME);
            newOutputMemSpace = ndOutputType.changeMemSpace(VPU::MemoryKind::CMX_NN);
            EXPECT_EQ(newOutputMemSpace.getMemSpace().getLeafName(), CMX_NAME);

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            const auto outputTile = ndOutputType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
            EXPECT_EQ(outputTile.getShape(), ShapeRef(tileShape));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            const auto paddedOutput = ndOutputType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
            EXPECT_EQ(paddedOutput.getShape(), ShapeRef({1, 16, 18, 18}));
        }
    }
}

TEST(MLIR_NDTypeInterface, UnrankedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    registry.insert<TestDialect>();
    registry.addTypeInterface<TestDialect, mlir::UnrankedTensorType, vpux::TensorNDTypeInterface>();

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: tensor<*xf16>) -> tensor<*xf16> {
                %0 = "test.UnrankedOp"(%arg0) : (tensor<*xf16>) -> (tensor<*xf16>)
                return %0 : tensor<*xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto testOp = mlir::dyn_cast<TestUnrankedOp>(op)) {
            const auto ndInputType = testOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
            EXPECT_TRUE(ndInputType != nullptr) << "Input is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndInputType.getShape(), ShapeRef({}));
            ASSERT_ANY_THROW(ndInputType.getMemShape());

            EXPECT_FALSE(ndInputType.hasRank());
            ASSERT_ANY_THROW(ndInputType.getRank());
            ASSERT_ANY_THROW(ndInputType.getNumElements());

            EXPECT_TRUE(ndInputType.getElementType().isa<mlir::Float16Type>());

            ASSERT_ANY_THROW(ndInputType.getDimsOrder());

            ASSERT_ANY_THROW(ndInputType.getMemSpace());
            ASSERT_ANY_THROW(ndInputType.getMemoryKind());

            ASSERT_ANY_THROW(ndInputType.getStrides());
            ASSERT_ANY_THROW(ndInputType.getMemStrides());

            EXPECT_EQ(ndInputType.getElemTypeSize().count(), 16);
            ASSERT_ANY_THROW(ndInputType.getTotalAllocSize());
            ASSERT_ANY_THROW(ndInputType.getCompactAllocSize());

            const SmallVector<int64_t> shape({1, 16, 64, 64});
            ASSERT_ANY_THROW(ndInputType.changeShape(ShapeRef(shape)));

            const auto elemType = mlir::Float32Type::get(op.getContext());
            const auto newInputElemType = ndInputType.changeElemType(elemType);
            EXPECT_TRUE(newInputElemType.getElementType().isa<mlir::Float32Type>());

            const auto dimsOrder = DimsOrder::NHWC;
            ASSERT_ANY_THROW(ndInputType.changeDimsOrder(dimsOrder));

            const auto memSpace = vpux::IndexedSymbolAttr::get(op.getContext(), CMX_NAME);
            ASSERT_ANY_THROW(ndInputType.changeMemSpace(memSpace));
            ASSERT_ANY_THROW(ndInputType.changeMemSpace(VPU::MemoryKind::CMX_NN));

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            ASSERT_ANY_THROW(ndInputType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape)));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            ASSERT_ANY_THROW(ndInputType.pad(ShapeRef(padBefore), ShapeRef(padAfter)));
        }
    }
}

TEST(MLIR_NDTypeInterface, MemRefType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: memref<1x16x32x32xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) -> memref<1x16x32x32xf16, #NHWC, @CMX_NN> {
                %cst_w = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<16x16x1x1xf16, #NHWC, @CMX_NN>
                %cst_wt = VPURT.DeclareBuffer "CMX_NN" <512> -> memref<16x1x1x4xsi32, @CMX_NN>
                VPURT.Task attributes {isTrailingSWLayer = false}  {
                    %16 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%arg0 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) weights(%cst_w : memref<16x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%cst_wt : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg0 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) parent_output(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) -> memref<1x16x32x32xf16, #NHWC, @CMX_NN> variants :  {
                    DPUTask {end = [31, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
                    DPUTask {end = [31, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
                    DPUTask {end = [31, 17, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
                    DPUTask {end = [31, 23, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 18, 0]}
                    DPUTask {end = [31, 31, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 24, 0]}
                    } PPE :  {
                    }
                }
                return %arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            auto wrappedTaskOp = taskOp.body().getBlocks().front().begin();
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp)) {
                // Input type
                const auto ndInputType = nceOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
                EXPECT_TRUE(ndInputType != nullptr) << "Input is not of vpux::NDTypeInterface type";

                EXPECT_EQ(ndInputType.getShape(), ShapeRef({1, 16, 32, 32}));
                EXPECT_EQ(ndInputType.getMemShape(), MemShape({1, 32, 32, 16}));

                EXPECT_TRUE(ndInputType.hasRank());
                EXPECT_EQ(ndInputType.getRank(), 4);
                EXPECT_EQ(ndInputType.getNumElements(), 16 * 32 * 32);

                EXPECT_TRUE(ndInputType.getElementType().isa<mlir::Float16Type>());

                EXPECT_EQ(ndInputType.getDimsOrder(), DimsOrder::NHWC);

                EXPECT_EQ(ndInputType.getMemSpace().getLeafName(), CMX_NAME);
                EXPECT_EQ(ndInputType.getMemoryKind(), VPU::MemoryKind::CMX_NN);

                const SmallVector<Bit> strides({262144_Bit, 16_Bit, 8192_Bit, 256_Bit});
                const SmallVector<Bit> memStrides({262144_Bit, 8192_Bit, 256_Bit, 16_Bit});
                EXPECT_EQ(ndInputType.getStrides().raw(), strides);
                EXPECT_EQ(ndInputType.getMemStrides().raw(), memStrides);

                EXPECT_EQ(ndInputType.getElemTypeSize().count(), 16);
                EXPECT_EQ(ndInputType.getTotalAllocSize().count(), 32768);
                EXPECT_EQ(ndInputType.getCompactAllocSize().count(), 32768);

                // Filter type
                const auto ndFilterType = nceOp.weights().getType().dyn_cast<vpux::NDTypeInterface>();
                EXPECT_TRUE(ndFilterType != nullptr) << "Filter is not of vpux::NDTypeInterface type";

                EXPECT_EQ(ndFilterType.getShape(), ShapeRef({16, 16, 1, 1}));
                EXPECT_EQ(ndFilterType.getMemShape(), MemShape({16, 1, 1, 16}));

                // Output type
                const auto ndOutputType = nceOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
                EXPECT_TRUE(ndOutputType != nullptr) << "Output is not of vpux::NDTypeInterface type";

                EXPECT_EQ(ndOutputType.getShape(), ShapeRef({1, 16, 32, 32}));
                const SmallVector<int64_t> shape({1, 16, 64, 64});
                const auto newOutputShape = ndOutputType.changeShape(ShapeRef(shape));
                EXPECT_EQ(newOutputShape.getShape(), ShapeRef(shape));

                EXPECT_TRUE(ndOutputType.getElementType().isa<mlir::Float16Type>());
                const auto elemType = mlir::Float32Type::get(op.getContext());
                const auto newOutputElemType = ndOutputType.changeElemType(elemType);
                EXPECT_TRUE(newOutputElemType.getElementType().isa<mlir::Float32Type>());

                EXPECT_EQ(ndOutputType.getDimsOrder(), DimsOrder::NHWC);
                const auto dimsOrder = DimsOrder::NCHW;
                const auto newOutputDimsOrder = ndOutputType.changeDimsOrder(dimsOrder);
                EXPECT_EQ(newOutputDimsOrder.getDimsOrder(), dimsOrder);

                EXPECT_EQ(ndOutputType.getMemSpace().getLeafName(), CMX_NAME);
                const auto memSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
                auto newOutputMemSpace = ndOutputType.changeMemSpace(memSpace);
                EXPECT_EQ(newOutputMemSpace.getMemSpace().getLeafName(), DDR_NAME);
                newOutputMemSpace = ndOutputType.changeMemSpace(VPU::MemoryKind::DDR);
                EXPECT_EQ(newOutputMemSpace.getMemSpace().getLeafName(), DDR_NAME);

                const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
                const SmallVector<int64_t> tileShape({1, 8, 32, 32});
                const auto outputTile = ndOutputType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
                EXPECT_EQ(outputTile.getShape(), ShapeRef(tileShape));

                const SmallVector<int64_t> padBefore({0, 0, 2, 2});
                const SmallVector<int64_t> padAfter({0, 0, 2, 2});
                const auto paddedOutput = ndOutputType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
                EXPECT_EQ(paddedOutput.getShape(), ShapeRef({1, 16, 36, 36}));
            }
        }
    }
}

TEST(MLIR_NDTypeInterface, UnrankedMemRefType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    registry.insert<TestDialect>();
    registry.addTypeInterface<TestDialect, mlir::UnrankedMemRefType, vpux::MemRefNDTypeInterface>();

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func @main(%arg0: memref<*xf16, @CMX_NN>, %arg1: memref<*xf16, @CMX_NN>) -> memref<*xf16, @CMX_NN> {
                %0 = "test.UnrankedOp"(%arg0, %arg1) : (memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) -> (memref<*xf16, @CMX_NN>)
                return %0 : memref<*xf16, @CMX_NN>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto testOp = mlir::dyn_cast<TestUnrankedOp>(op)) {
            const auto ndInputType = testOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
            EXPECT_TRUE(ndInputType != nullptr) << "Input is not of vpux::NDTypeInterface type";

            EXPECT_EQ(ndInputType.getShape(), ShapeRef({}));
            ASSERT_ANY_THROW(ndInputType.getMemShape());

            EXPECT_FALSE(ndInputType.hasRank());
            ASSERT_ANY_THROW(ndInputType.getRank());
            ASSERT_ANY_THROW(ndInputType.getNumElements());

            EXPECT_TRUE(ndInputType.getElementType().isa<mlir::Float16Type>());

            ASSERT_ANY_THROW(ndInputType.getDimsOrder());

            EXPECT_EQ(ndInputType.getMemSpace().getLeafName(), CMX_NAME);
            EXPECT_EQ(ndInputType.getMemoryKind(), VPU::MemoryKind::CMX_NN);

            ASSERT_ANY_THROW(ndInputType.getStrides());
            ASSERT_ANY_THROW(ndInputType.getMemStrides());

            EXPECT_EQ(ndInputType.getElemTypeSize().count(), 16);
            ASSERT_ANY_THROW(ndInputType.getTotalAllocSize());
            ASSERT_ANY_THROW(ndInputType.getCompactAllocSize());

            const SmallVector<int64_t> shape({1, 16, 64, 64});
            ASSERT_ANY_THROW(ndInputType.changeShape(ShapeRef(shape)));

            const auto elemType = mlir::Float32Type::get(op.getContext());
            const auto newInputElemType = ndInputType.changeElemType(elemType);
            EXPECT_TRUE(newInputElemType.getElementType().isa<mlir::Float32Type>());

            const auto dimsOrder = DimsOrder::NHWC;
            ASSERT_ANY_THROW(ndInputType.changeDimsOrder(dimsOrder));

            const auto memSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
            auto newInputMemSpace = ndInputType.changeMemSpace(memSpace);
            EXPECT_EQ(newInputMemSpace.getMemSpace().getLeafName(), DDR_NAME);
            newInputMemSpace = ndInputType.changeMemSpace(VPU::MemoryKind::DDR);
            EXPECT_EQ(newInputMemSpace.getMemSpace().getLeafName(), DDR_NAME);

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            ASSERT_ANY_THROW(ndInputType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape)));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            ASSERT_ANY_THROW(ndInputType.pad(ShapeRef(padBefore), ShapeRef(padAfter)));
        }
    }
}

TEST(MLIR_NDTypeInterface, SparseBufferType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: memref<1x16x32x32xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) -> memref<1x16x32x32xf16, #NHWC, @CMX_NN> {
                %cst_w = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<16x16x1x1xf16, #NHWC, @CMX_NN>
                %cst_sm = VPURT.DeclareBuffer "CMX_NN" <512> -> memref<16x16x1x1xi1, #NHWC, @CMX_NN>
                %cst_sparse = VPURT.DeclareSparseBuffer %cst_w : memref<16x16x1x1xf16, #NHWC, @CMX_NN>, %cst_sm : memref<16x16x1x1xi1, #NHWC, @CMX_NN> -> !VPURT.SparseBuffer<data=memref<16x16x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<16x16x1x1xi1, #NHWC, @CMX_NN>>
                %cst_wt = VPURT.DeclareBuffer "CMX_NN" <544> -> memref<16x1x1x4xsi32, @CMX_NN>
                VPURT.Task attributes {isTrailingSWLayer = false}  {
                    %16 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%arg0 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) weights(%cst_sparse : !VPURT.SparseBuffer<data=memref<16x16x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<16x16x1x1xi1, #NHWC, @CMX_NN>>) weight_table(%cst_wt : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg0 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) parent_output(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) -> memref<1x16x32x32xf16, #NHWC, @CMX_NN> variants :  {
                    DPUTask {end = [31, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
                    DPUTask {end = [31, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
                    DPUTask {end = [31, 17, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
                    DPUTask {end = [31, 23, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 18, 0]}
                    DPUTask {end = [31, 31, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 24, 0]}
                    } PPE :  {
                    }
                }
                return %arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>
            }
        }
    )";

    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(VPU::ArchKind::KMB, VPU::CompilationMode::DefaultHW, None, Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            auto wrappedTaskOp = taskOp.body().getBlocks().front().begin();
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp)) {
                // Sparse filter type
                const auto ndFilterType = nceOp.weights().getType().dyn_cast<vpux::NDTypeInterface>();
                EXPECT_TRUE(ndFilterType != nullptr) << "Filter is not of vpux::NDTypeInterface type";

                EXPECT_EQ(ndFilterType.getShape(), ShapeRef({16, 16, 1, 1}));
                EXPECT_EQ(ndFilterType.getMemShape(), MemShape({16, 1, 1, 16}));

                EXPECT_TRUE(ndFilterType.hasRank());
                EXPECT_EQ(ndFilterType.getRank(), 4);
                EXPECT_EQ(ndFilterType.getNumElements(), 16 * 16);

                EXPECT_TRUE(ndFilterType.getElementType().isa<mlir::Float16Type>());

                EXPECT_EQ(ndFilterType.getDimsOrder(), DimsOrder::NHWC);

                EXPECT_EQ(ndFilterType.getMemSpace().getLeafName(), CMX_NAME);
                EXPECT_EQ(ndFilterType.getMemoryKind(), VPU::MemoryKind::CMX_NN);

                const SmallVector<Bit> strides({256_Bit, 16_Bit, 256_Bit, 256_Bit});
                const SmallVector<Bit> memStrides({256_Bit, 256_Bit, 256_Bit, 16_Bit});
                EXPECT_EQ(ndFilterType.getStrides().raw(), strides);
                EXPECT_EQ(ndFilterType.getMemStrides().raw(), memStrides);

                EXPECT_EQ(ndFilterType.getElemTypeSize().count(), 16);
                EXPECT_EQ(ndFilterType.getTotalAllocSize().count(), 544);
                EXPECT_EQ(ndFilterType.getCompactAllocSize().count(), 544);

                EXPECT_EQ(ndFilterType.getShape(), ShapeRef({16, 16, 1, 1}));
                const SmallVector<int64_t> shape({8, 32, 1, 1});
                const auto newFilterShape = ndFilterType.changeShape(ShapeRef(shape));
                EXPECT_EQ(newFilterShape.getShape(), ShapeRef(shape));

                EXPECT_TRUE(ndFilterType.getElementType().isa<mlir::Float16Type>());
                const auto elemType = mlir::Float32Type::get(op.getContext());
                const auto newFilterElemType = ndFilterType.changeElemType(elemType);
                EXPECT_TRUE(newFilterElemType.getElementType().isa<mlir::Float32Type>());

                EXPECT_EQ(ndFilterType.getDimsOrder(), DimsOrder::NHWC);
                const auto dimsOrder = DimsOrder::NCHW;
                const auto newFilterDimsOrder = ndFilterType.changeDimsOrder(dimsOrder);
                EXPECT_EQ(newFilterDimsOrder.getDimsOrder(), dimsOrder);

                EXPECT_EQ(ndFilterType.getMemSpace().getLeafName(), CMX_NAME);
                const auto memSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
                auto newFilterMemSpace = ndFilterType.changeMemSpace(memSpace);
                EXPECT_EQ(newFilterMemSpace.getMemSpace().getLeafName(), DDR_NAME);
                newFilterMemSpace = ndFilterType.changeMemSpace(VPU::MemoryKind::DDR);
                EXPECT_EQ(newFilterMemSpace.getMemSpace().getLeafName(), DDR_NAME);
            }
        }
    }
}
