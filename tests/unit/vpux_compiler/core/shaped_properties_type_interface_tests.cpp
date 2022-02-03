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

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

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

TEST(MLIR_ShapedPropertiesTypeInterface, RankedTensorType) {
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
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto convOp = mlir::dyn_cast<vpux::IE::ConvolutionOp>(op)) {
            // Input type
            const auto shapedInputType = convOp.input().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedInputType != nullptr) << "Input is not of vpux::ShapedPropertiesTypeInterface type";

            EXPECT_EQ(shapedInputType.getShape(), vpux::ShapeRef({1, 16, 32, 32}));
            EXPECT_EQ(shapedInputType.getMemShape(), vpux::MemShape({1, 16, 32, 32}));

            EXPECT_TRUE(shapedInputType.hasRank());
            EXPECT_EQ(shapedInputType.getRank(), 4);
            EXPECT_EQ(shapedInputType.getNumElements(), 16*32*32);

            EXPECT_TRUE(shapedInputType.getElementType().isa<mlir::Float16Type>());

            EXPECT_EQ(shapedInputType.getDimsOrder(), vpux::DimsOrder::NCHW);

            EXPECT_EQ(shapedInputType.getMemSpace(), nullptr);
            EXPECT_EQ(shapedInputType.getMemoryKind(), vpux::VPU::MemoryKind::DDR);

            const SmallVector<vpux::Bit> strides({262144_Bit, 16384_Bit, 512_Bit, 16_Bit});
            EXPECT_EQ(shapedInputType.getStrides().raw(), strides);
            EXPECT_EQ(shapedInputType.getMemStrides().raw(), strides);

            EXPECT_EQ(shapedInputType.getElemTypeSize().count(), 16);
            EXPECT_EQ(shapedInputType.getTotalAllocSize().count(), 32768);
            EXPECT_EQ(shapedInputType.getCompactAllocSize().count(), 32768);

            // Filter type
            const auto shapedFilterType = convOp.filter().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedFilterType != nullptr) << "Filter is not of vpux::ShapedPropertiesTypeInterface type";

            EXPECT_EQ(shapedFilterType.getShape(), vpux::ShapeRef({16, 16, 1, 1}));
            EXPECT_EQ(shapedFilterType.getMemShape(), vpux::MemShape({16, 1, 1, 16}));

            EXPECT_EQ(shapedFilterType.getDimsOrder(), vpux::DimsOrder::NHWC);

            // Output type
            const auto shapedOutputType = convOp.output().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedOutputType != nullptr) << "Output is not of vpux::ShapedPropertiesTypeInterface type";

            EXPECT_EQ(shapedOutputType.getShape(), vpux::ShapeRef({1, 16, 16, 16}));
            const SmallVector<int64_t> newShape({1, 16, 64, 64});
            const auto newOutputTypeShape = shapedOutputType.changeShape(vpux::ShapeRef(newShape));
            EXPECT_EQ(newOutputTypeShape.getShape(), vpux::ShapeRef(newShape));

            EXPECT_TRUE(shapedOutputType.getElementType().isa<mlir::Float16Type>());
            const auto newElemType = mlir::Float32Type::get(op.getContext());
            const auto newOutputTypeElemType = shapedOutputType.changeElemType(newElemType);
            EXPECT_TRUE(newOutputTypeElemType.getElementType().isa<mlir::Float32Type>());

            EXPECT_EQ(shapedOutputType.getDimsOrder(), DimsOrder::NCHW);
            const auto newDimsOrder = DimsOrder::NHWC;
            const auto newOutputTypeDimsOrder = shapedOutputType.changeDimsOrder(newDimsOrder);
            EXPECT_EQ(newOutputTypeDimsOrder.getDimsOrder(), newDimsOrder);

            EXPECT_EQ(shapedOutputType.getMemSpace(), nullptr);
            const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), CMX_NAME);
            const auto newOutputTypeMemSpace = shapedOutputType.changeMemSpace(newMemSpace);
            EXPECT_EQ(newOutputTypeMemSpace.getMemSpace().getLeafName(), CMX_NAME);

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            const auto outputTile = shapedOutputType.extractDenseTile(vpux::ShapeRef(tileOffsets), vpux::ShapeRef(tileShape));
            EXPECT_EQ(outputTile.getShape(), vpux::ShapeRef(tileShape));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            const auto paddedOutput = shapedOutputType.pad(vpux::ShapeRef(padBefore), vpux::ShapeRef(padAfter));
            EXPECT_EQ(paddedOutput.getShape(), vpux::ShapeRef({1, 16, 18, 18}));
        }
    }
}

TEST(MLIR_ShapedPropertiesTypeInterface, UnrankedTensorType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    registry.insert<TestDialect>();
    registry.addTypeInterface<TestDialect, mlir::UnrankedTensorType, vpux::TensorPropertiesTypeInterface>();

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
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto testOp = mlir::dyn_cast<TestUnrankedOp>(op)) {
            const auto shapedInputType = testOp->getOperand(0).getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedInputType != nullptr) << "Input is not of vpux::ShapedPropertiesTypeInterface type";

            EXPECT_EQ(shapedInputType.getShape(), vpux::ShapeRef({}));
            ASSERT_ANY_THROW(shapedInputType.getMemShape());

            EXPECT_FALSE(shapedInputType.hasRank());
            ASSERT_ANY_THROW(shapedInputType.getRank());
            ASSERT_ANY_THROW(shapedInputType.getNumElements());

            EXPECT_TRUE(shapedInputType.getElementType().isa<mlir::Float16Type>());

            ASSERT_ANY_THROW(shapedInputType.getDimsOrder());

            ASSERT_ANY_THROW(shapedInputType.getMemSpace());
            ASSERT_ANY_THROW(shapedInputType.getMemoryKind());

            ASSERT_ANY_THROW(shapedInputType.getStrides());
            ASSERT_ANY_THROW(shapedInputType.getMemStrides());

            EXPECT_EQ(shapedInputType.getElemTypeSize().count(), 16);
            ASSERT_ANY_THROW(shapedInputType.getTotalAllocSize());
            ASSERT_ANY_THROW(shapedInputType.getCompactAllocSize());

            const SmallVector<int64_t> newShape({1, 16, 64, 64});
            ASSERT_ANY_THROW(shapedInputType.changeShape(vpux::ShapeRef(newShape)));

            const auto newElemType = mlir::Float32Type::get(op.getContext());
            const auto newInputTypeElemType = shapedInputType.changeElemType(newElemType);
            EXPECT_TRUE(newInputTypeElemType.getElementType().isa<mlir::Float32Type>());

            const auto newDimsOrder = DimsOrder::NHWC;
            ASSERT_ANY_THROW(shapedInputType.changeDimsOrder(newDimsOrder));

            const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), CMX_NAME);
            ASSERT_ANY_THROW(shapedInputType.changeMemSpace(newMemSpace));

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            ASSERT_ANY_THROW(shapedInputType.extractDenseTile(vpux::ShapeRef(tileOffsets), vpux::ShapeRef(tileShape)));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            ASSERT_ANY_THROW(shapedInputType.pad(vpux::ShapeRef(padBefore), vpux::ShapeRef(padAfter)));
        }
    }
}

TEST(MLIR_ShapedPropertiesTypeInterface, MemRefType) {
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
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<vpux::VPURT::TaskOp>(op)) {
            auto wrappedTaskOp = taskOp.body().getBlocks().front().begin();
            if (auto nceOp = mlir::dyn_cast<vpux::VPUIP::NCEClusterTaskOp>(wrappedTaskOp)) {
                // Input type
                const auto shapedInputType = nceOp.input().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
                EXPECT_TRUE(shapedInputType != nullptr) << "Input is not of vpux::ShapedPropertiesTypeInterface type";

                EXPECT_EQ(shapedInputType.getShape(), vpux::ShapeRef({1, 16, 32, 32}));
                EXPECT_EQ(shapedInputType.getMemShape(), vpux::MemShape({1, 32, 32, 16}));

                EXPECT_TRUE(shapedInputType.hasRank());
                EXPECT_EQ(shapedInputType.getRank(), 4);
                EXPECT_EQ(shapedInputType.getNumElements(), 16*32*32);

                EXPECT_TRUE(shapedInputType.getElementType().isa<mlir::Float16Type>());

                EXPECT_EQ(shapedInputType.getDimsOrder(), vpux::DimsOrder::NHWC);

                EXPECT_EQ(shapedInputType.getMemSpace().getLeafName(), CMX_NAME);
                EXPECT_EQ(shapedInputType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

                const SmallVector<vpux::Bit> strides({262144_Bit, 16_Bit, 8192_Bit, 256_Bit});
                const SmallVector<vpux::Bit> memStrides({262144_Bit, 8192_Bit, 256_Bit, 16_Bit});
                EXPECT_EQ(shapedInputType.getStrides().raw(), strides);
                EXPECT_EQ(shapedInputType.getMemStrides().raw(), memStrides);

                EXPECT_EQ(shapedInputType.getElemTypeSize().count(), 16);
                EXPECT_EQ(shapedInputType.getTotalAllocSize().count(), 32768);
                EXPECT_EQ(shapedInputType.getCompactAllocSize().count(), 32768);

                // Filter type
                const auto shapedFilterType = nceOp.weights().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
                EXPECT_TRUE(shapedFilterType != nullptr) << "Filter is not of vpux::ShapedPropertiesTypeInterface type";

                EXPECT_EQ(shapedFilterType.getShape(), vpux::ShapeRef({16, 16, 1, 1}));
                EXPECT_EQ(shapedFilterType.getMemShape(), vpux::MemShape({16, 1, 1, 16}));

                // Output type
                const auto shapedOutputType = nceOp.output().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
                EXPECT_TRUE(shapedOutputType != nullptr) << "Output is not of vpux::ShapedPropertiesTypeInterface type";

                EXPECT_EQ(shapedOutputType.getShape(), vpux::ShapeRef({1, 16, 32, 32}));
                const SmallVector<int64_t> newShape({1, 16, 64, 64});
                const auto newOutputTypeShape = shapedOutputType.changeShape(vpux::ShapeRef(newShape));
                EXPECT_EQ(newOutputTypeShape.getShape(), vpux::ShapeRef(newShape));

                EXPECT_TRUE(shapedOutputType.getElementType().isa<mlir::Float16Type>());
                const auto newElemType = mlir::Float32Type::get(op.getContext());
                const auto newOutputTypeElemType = shapedOutputType.changeElemType(newElemType);
                EXPECT_TRUE(newOutputTypeElemType.getElementType().isa<mlir::Float32Type>());

                EXPECT_EQ(shapedOutputType.getDimsOrder(), DimsOrder::NHWC);
                const auto newDimsOrder = DimsOrder::NCHW;
                const auto newOutputTypeDimsOrder = shapedOutputType.changeDimsOrder(newDimsOrder);
                EXPECT_EQ(newOutputTypeDimsOrder.getDimsOrder(), newDimsOrder);

                EXPECT_EQ(shapedOutputType.getMemSpace().getLeafName(), CMX_NAME);
                const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
                const auto newOutputTypeMemSpace = shapedOutputType.changeMemSpace(newMemSpace);
                EXPECT_EQ(newOutputTypeMemSpace.getMemSpace().getLeafName(), DDR_NAME);

                const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
                const SmallVector<int64_t> tileShape({1, 8, 32, 32});
                const auto outputTile =
                        shapedOutputType.extractDenseTile(vpux::ShapeRef(tileOffsets), vpux::ShapeRef(tileShape));
                EXPECT_EQ(outputTile.getShape(), vpux::ShapeRef(tileShape));

                const SmallVector<int64_t> padBefore({0, 0, 2, 2});
                const SmallVector<int64_t> padAfter({0, 0, 2, 2});
                const auto paddedOutput = shapedOutputType.pad(vpux::ShapeRef(padBefore), vpux::ShapeRef(padAfter));
                EXPECT_EQ(paddedOutput.getShape(), vpux::ShapeRef({1, 16, 36, 36}));
            }
        }
    }
}


TEST(MLIR_ShapedPropertiesTypeInterface, UnrankedMemRefType) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    registry.insert<TestDialect>();
    registry.addTypeInterface<TestDialect, mlir::UnrankedMemRefType, vpux::MemRefPropertiesTypeInterface>();

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
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto testOp = mlir::dyn_cast<TestUnrankedOp>(op)) {
            const auto shapedInputType = testOp->getOperand(0).getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
            EXPECT_TRUE(shapedInputType != nullptr) << "Input is not of vpux::ShapedPropertiesTypeInterface type";

            EXPECT_EQ(shapedInputType.getShape(), vpux::ShapeRef({}));
            ASSERT_ANY_THROW(shapedInputType.getMemShape());

            EXPECT_FALSE(shapedInputType.hasRank());
            ASSERT_ANY_THROW(shapedInputType.getRank());
            ASSERT_ANY_THROW(shapedInputType.getNumElements());

            EXPECT_TRUE(shapedInputType.getElementType().isa<mlir::Float16Type>());

            ASSERT_ANY_THROW(shapedInputType.getDimsOrder());

            EXPECT_EQ(shapedInputType.getMemSpace().getLeafName(), CMX_NAME);
            EXPECT_EQ(shapedInputType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

            ASSERT_ANY_THROW(shapedInputType.getStrides());
            ASSERT_ANY_THROW(shapedInputType.getMemStrides());

            EXPECT_EQ(shapedInputType.getElemTypeSize().count(), 16);
            ASSERT_ANY_THROW(shapedInputType.getTotalAllocSize());
            ASSERT_ANY_THROW(shapedInputType.getCompactAllocSize());

            const SmallVector<int64_t> newShape({1, 16, 64, 64});
            ASSERT_ANY_THROW(shapedInputType.changeShape(vpux::ShapeRef(newShape)));

            const auto newElemType = mlir::Float32Type::get(op.getContext());
            const auto newInputTypeElemType = shapedInputType.changeElemType(newElemType);
            EXPECT_TRUE(newInputTypeElemType.getElementType().isa<mlir::Float32Type>());

            const auto newDimsOrder = DimsOrder::NHWC;
            ASSERT_ANY_THROW(shapedInputType.changeDimsOrder(newDimsOrder));

            const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
            const auto newInputTypeMemSpace = shapedInputType.changeMemSpace(newMemSpace);
            EXPECT_EQ(newInputTypeMemSpace.getMemSpace().getLeafName(), DDR_NAME);

            const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
            const SmallVector<int64_t> tileShape({1, 8, 16, 16});
            ASSERT_ANY_THROW(shapedInputType.extractDenseTile(vpux::ShapeRef(tileOffsets), vpux::ShapeRef(tileShape)));

            const SmallVector<int64_t> padBefore({0, 0, 1, 1});
            const SmallVector<int64_t> padAfter({0, 0, 1, 1});
            ASSERT_ANY_THROW(shapedInputType.pad(vpux::ShapeRef(padBefore), vpux::ShapeRef(padAfter)));
        }
    }
}

TEST(MLIR_ShapedPropertiesTypeInterface, SparseBufferType) {
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
    pm.addPass(vpux::VPU::createInitCompilerPass(vpux::VPU::ArchKind::KMB, vpux::VPU::CompilationMode::DefaultHW,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<vpux::VPURT::TaskOp>(op)) {
            auto wrappedTaskOp = taskOp.body().getBlocks().front().begin();
            if (auto nceOp = mlir::dyn_cast<vpux::VPUIP::NCEClusterTaskOp>(wrappedTaskOp)) {
                // Sparse filter type
                const auto shapedFilterType = nceOp.weights().getType().dyn_cast<vpux::ShapedPropertiesTypeInterface>();
                EXPECT_TRUE(shapedFilterType != nullptr) << "Filter is not of vpux::ShapedPropertiesTypeInterface type";

                EXPECT_EQ(shapedFilterType.getShape(), vpux::ShapeRef({16, 16, 1, 1}));
                EXPECT_EQ(shapedFilterType.getMemShape(), vpux::MemShape({16, 1, 1, 16}));

                EXPECT_TRUE(shapedFilterType.hasRank());
                EXPECT_EQ(shapedFilterType.getRank(), 4);
                EXPECT_EQ(shapedFilterType.getNumElements(), 16*16);

                EXPECT_TRUE(shapedFilterType.getElementType().isa<mlir::Float16Type>());

                EXPECT_EQ(shapedFilterType.getDimsOrder(), vpux::DimsOrder::NHWC);

                EXPECT_EQ(shapedFilterType.getMemSpace().getLeafName(), CMX_NAME);
                EXPECT_EQ(shapedFilterType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

                const SmallVector<vpux::Bit> strides({256_Bit, 16_Bit, 256_Bit, 256_Bit});
                const SmallVector<vpux::Bit> memStrides({256_Bit, 256_Bit, 256_Bit, 16_Bit});
                EXPECT_EQ(shapedFilterType.getStrides().raw(), strides);
                EXPECT_EQ(shapedFilterType.getMemStrides().raw(), memStrides);

                EXPECT_EQ(shapedFilterType.getElemTypeSize().count(), 16);
                EXPECT_EQ(shapedFilterType.getTotalAllocSize().count(), 544);
                EXPECT_EQ(shapedFilterType.getCompactAllocSize().count(), 544);

                EXPECT_EQ(shapedFilterType.getShape(), vpux::ShapeRef({16, 16, 1, 1}));
                const SmallVector<int64_t> newShape({8, 32, 1, 1});
                const auto newFilterTypeShape = shapedFilterType.changeShape(vpux::ShapeRef(newShape));
                EXPECT_EQ(newFilterTypeShape.getShape(), vpux::ShapeRef(newShape));

                EXPECT_TRUE(shapedFilterType.getElementType().isa<mlir::Float16Type>());
                const auto newElemType = mlir::Float32Type::get(op.getContext());
                const auto newFilterTypeElemType = shapedFilterType.changeElemType(newElemType);
                EXPECT_TRUE(newFilterTypeElemType.getElementType().isa<mlir::Float32Type>());

                EXPECT_EQ(shapedFilterType.getDimsOrder(), DimsOrder::NHWC);
                const auto newDimsOrder = DimsOrder::NCHW;
                const auto newFilterTypeDimsOrder = shapedFilterType.changeDimsOrder(newDimsOrder);
                EXPECT_EQ(newFilterTypeDimsOrder.getDimsOrder(), newDimsOrder);

                EXPECT_EQ(shapedFilterType.getMemSpace().getLeafName(), CMX_NAME);
                const auto newMemSpace = vpux::IndexedSymbolAttr::get(op.getContext(), DDR_NAME);
                const auto newFilterTypeMemSpace = shapedFilterType.changeMemSpace(newMemSpace);
                EXPECT_EQ(newFilterTypeMemSpace.getMemSpace().getLeafName(), DDR_NAME);
            }
        }
    }
}
