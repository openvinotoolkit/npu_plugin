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

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"

namespace vpux {
namespace hwtest {
using llvm::ArrayRef;

// NumericsBench padding definition
// NB ref: NumericsBench/operators/op_utils/sliding_window.py#L47
// IERT Conv pads_begin/pads_end ref: kmb-plugin/src/vpux_compiler/src/core/tiling.cpp#L308
#define PAD_NB_TOP 0
#define PAD_NB_LEFT 1
#define PAD_NB_BOTTOM 2
#define PAD_NB_RIGHT 3

// NCETask padding definition
// IERT::ConvolutionOp -> VPUIP::NCEInvariant ref:
// kmb-plugin/src/vpux_compiler/src/conversion/passes/convert_to_nce_ops.cpp#L185
#define PAD_NCETASK_LEFT 0
#define PAD_NCETASK_RIGHT 1
#define PAD_NCETASK_TOP 2
#define PAD_NCETASK_BOTTOM 3

mlir::DenseElementsAttr generateWeights(ArrayRef<int64_t> wt_shape, mlir::Type dtype, mlir::MLIRContext* ctx,
                                        const char* weight_file_name);
void computeQuantMultShift(float scale, unsigned& shift, unsigned& mult);
size_t calcWeightsTableMultShift(const nb::TestCaseJsonDescriptor& testDesc, mlir::MemRefType input,
                                 mlir::MemRefType output, mlir::MemRefType weights);
std::vector<int32_t> generateWeightsTablesValues(const nb::TestCaseJsonDescriptor& testDesc, size_t weights_offset,
                                                 mlir::MemRefType input, mlir::MemRefType output,
                                                 mlir::MemRefType weights);
uint16_t getWindowSize(mlir::OpBuilder builder, uint16_t kx, uint16_t sx, mlir::Type dataType);
std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels);
mlir::DenseElementsAttr getactivationWindow(mlir::OpBuilder builder, const std::vector<int64_t>& filter_size,
                                            const std::vector<int64_t>& strides, int64_t outputChannels,
                                            int64_t inputChannels, mlir::Type dtype,
                                            SmallVector<int64_t>& sparsity_shape,
                                            mlir::IntegerAttr& activationChannelLength, bool isOutFloat, bool isPool,
                                            bool isDepthWiseConv);
unsigned round_up(unsigned x, unsigned mult);
std::vector<int32_t> generateWeightsTablesValuesForMaxPool(const nb::TestCaseJsonDescriptor& testDesc,
                                                           mlir::MemRefType input, mlir::MemRefType output,
                                                           mlir::MemRefType actWindow_cmx_type, std::size_t offset,
                                                           ArrayRef<int64_t> wtTbl_data_shape);

std::vector<int32_t> generateWeightsTablesValuesWithSparsity(const nb::TestCaseJsonDescriptor& testDesc,
                                                             mlir::MemRefType input, mlir::MemRefType output,
                                                             mlir::MemRefType weights,
                                                             mlir::MemRefType actWindow_cmx_type, std::size_t offset,
                                                             ArrayRef<int64_t> wtTbl_data_shape, size_t weights_offset);
SmallVector<int64_t> getWeightsPaddedShape(SmallVector<int64_t> wt_shape, bool isDepthwiseConv);
mlir::DenseElementsAttr generateDWConvWeightsForAvgPool(ArrayRef<int64_t> wt_shape, mlir::Type dtype, double scaleVal,
                                                        mlir::MLIRContext* ctx);
mlir::DenseElementsAttr generateZeroPadForEltwiseMultWeights(ArrayRef<int64_t> wt_shape_padded, mlir::Type dtype,
                                                             mlir::MLIRContext* ctx);
mlir::Type convertToMLIRType(mlir::OpBuilder builder, nb::DType dtype);
mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input);
mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output);
mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight);
VPUIP::PPELayerType getPPELayerFromConfig(nb::ActivationLayer activation);
int32_t computeclampLow(nb::InputLayer input, nb::OutputLayer output, bool flexarbINT8, bool isMaxpool,
                        nb::ActivationLayer activation);
int32_t computeclampHigh(nb::InputLayer input, nb::OutputLayer output, bool flexarbINT8, bool isMaxpool,
                         nb::ActivationLayer activation);

void calculateppeParams(const nb::TestCaseJsonDescriptor& testDesc, int32_t& clampLow, int32_t& clamHigh,
                        int32_t& lreluMult, uint32_t& lreluShift);

void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);

void buildDWConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                 Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);

void buildSimpleZMajorConvActivation(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                     mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                     mlir::Type outputType);

void buildEltwiseAdd(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);

void buildEltwiseMultWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                mlir::Type outputType);

void buildMaxpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type input0Type, mlir::Type outputType);

void buildActKernelTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log);

void buildPipeline(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType, bool isSequential);

void buildRaceConditionDMATest(const nb::TestCaseJsonDescriptor&, mlir::ModuleOp module, mlir::OpBuilder builder,
                               Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildRaceConditionDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                               mlir::Type outputType);
void buildAvgpoolWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType);

std::size_t totalTensorSize(llvm::ArrayRef<std::int64_t> shape, mlir::Type elementtype);
std::vector<int32_t> getInstructionListVals(nb::ActivationType pwlType,
                                            llvm::ArrayRef<int64_t> instructionList_data_shape);

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs);

mlir::MemRefType getMemRefType(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation, SmallVector<int64_t> shape,
                               mlir::Type type, SmallVector<mlir::AffineMap> affineMaps);

vpux::VPUIP::DeclareTensorOp createDeclareTensorOp(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation,
                                                   SmallVector<int64_t> shape, mlir::Type type,
                                                   SmallVector<mlir::AffineMap> affineMaps, int locale, int offset);

mlir::OpResult getTensorResult(VPUIP::DeclareTensorOp op);

mlir::OpResult getConstResult(vpux::Const::DeclareOp op);

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       llvm::SmallVector<int64_t> output_shape, std::vector<int64_t> padding_vec);

std::vector<int64_t> convertNBPadtoNCETaskPad(std::array<int64_t, 4>& nb_pad);

}  // namespace hwtest
}  // namespace vpux
