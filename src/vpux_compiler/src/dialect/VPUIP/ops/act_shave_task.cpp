//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/movitools/movitools.h"

using namespace vpux;
using namespace mlir;

namespace vpux {
namespace VPUIP {

VPUIP::BlobWriter::SpecificTask ACTShaveTaskOp::serialize(VPUIP::BlobWriter& writer) {
    // Kernel binary
    SymbolRefAttr kernelRef = kernel();
    auto module = (*this)->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<FuncOp>(kernelRef);
    VPUX_THROW_UNLESS(func != nullptr, "Could not resolve kernel symbol reference '{0}'", kernelRef);
    movitools::MoviCompileParams params = {
            /*cpu=*/"3010xx",
            /*moviCompile=*/"linux64/bin/moviCompile",
            /*mdkLinker=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld",
            /*mdkLibDir=*/"common/moviCompile/lib/30xxxx-leon",
            /*mdkLibs=*/
            {
                    "mlibcxx.a",
                    "mlibneon.a",
                    "mlibVecUtils.a",
                    "mlibm.a",
                    "mlibc_lite.a",
                    "mlibc_lite_lgpl.a",
                    "mlibcrt.a",
            },
    };

    //    const auto axisDim = getAxisDim();

    flatbuffers::Offset<MVCNN::BinaryData> elfBinary = generateKernelForACTShave(func, params, writer);


    // TODO: figure out current approach seems based on some structure
    /*MVCNN::ActKernelParamsBuilder builder(writer);

    builder.add_axis(checked_cast<uint32_t>(axisDim.ind()));
    const auto paramsOff = builder.Finish();*/
    MVCNN::KernelDataBuilder shaveKernelBuilder(writer);

    return writer.createACTShaveTask(*this, {shaveKernelBuilder.Finish(), MVCNN::ActKernelType_KERNEL});
}
/*
mlir::Operation* vpux::VPUIP::BlobReader::parseActShave(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
        ArrayRef<mlir::Value> outputs, const MVCNN::ActShaveTask* task) {
VPUX_THROW_UNLESS(inputs.size() == 1, "UPASoftMax supports only 1 input, got {0}", inputs.size());
VPUX_THROW_UNLESS(outputs.size() == 1, "UPASoftMax supports only 1 output, got {0}", outputs.size());
const auto params = task->softLayerParams_as_SoftmaxParams();
const auto axis = getInt32Attr(_ctx, params->axis());
return builder.create<VPUIP::SoftMaxUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], axis);
}*/

}  // namespace VPUIP
}  // namespace vpux