//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2ITaskOp::inferReturnTypes(mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    M2ITaskOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto inElemType = inType.getElementType();

    SmallVector<int64_t> outShape(4);
    auto N = inShape[Dims4D::Act::N.ind()];
    auto C = inShape[Dims4D::Act::C.ind()];
    auto H = inShape[Dims4D::Act::H.ind()];
    auto W = inShape[Dims4D::Act::W.ind()];

    if (op.do_csc()) {  // N,H,W,C input
        // TBD: check {input:NV12/I420, output:RGB} as for RGB->NV12 transform will differ
        H = inShape[1];
        W = inShape[2];
        C = inShape[3];
        // Y,UV (1 or 2 plane configs) are expected to have C = 1
        if (C != 1) {
            return errorAt(loc, "Incorrect number of channels: expecting 1, got '{0}'", C);
        }
        // OK for NV12/I420 -> RGB/BGR :
        // input Height is big enough to include Chroma, so lower for RGB output
        H = H * 2 / 3;
        C = 3;  //(3 ch: RGB)
    }

    // doing Resize ?
    if (op.sizes().hasValue()) {
        // Note: limited to 'shape_calculation_mode = sizes'
        const auto outSize = parseIntArrayAttr<int64_t>(op.sizes().getValue());
        const auto outAxes = parseIntArrayAttr<int64_t>(op.axes().getValue());

        // last 2 dims of 'axes' are always H,W (in either NCHW or NHWC cases)
        H = outSize[outAxes.size() - 2];
        W = outSize[outAxes.size() - 1];
        if (outAxes[1] != 3) {
            C = op.do_csc() ? 3 : inShape[3];
        }
    }

    auto oFmt = op.outFmt();
    mlir::Type outElemType = inElemType;

    if ((oFmt == M2iColorFmt::PL_FP16_RGB) || (oFmt == M2iColorFmt::PL_FP16_BGR) || (oFmt == M2iColorFmt::PL_RGB24) ||
        (oFmt == M2iColorFmt::PL_BGR24)) {
        outShape[0] = N;
        outShape[1] = C;
        outShape[2] = H;
        outShape[3] = W;
    } else if ((oFmt == M2iColorFmt::IL_RGB888) || (oFmt == M2iColorFmt::IL_BGR888)) {
        outShape[0] = N;
        outShape[1] = H;
        outShape[2] = W;
        outShape[3] = C;
    } else {
        VPUX_THROW("M2iTask unsupported out format '{0}'", oFmt);
    }

    if ((oFmt == M2iColorFmt::PL_FP16_RGB) || (oFmt == M2iColorFmt::PL_FP16_BGR)) {
        outElemType = mlir::Float16Type::get(ctx);
    }

    const auto outType = inType.changeElemType(outElemType).changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::VPU::M2ITaskOp::serialize(EMU::BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    EMU::BlobWriter::Vector<uint16_t> serializedCoefs;

    if (norm().hasValue()) {
        const auto coefs = parseFPArrayAttr<double>(norm().getValue());
        serializedCoefs = getVecFP16(coefs);
    }

    const auto getTensorCb = [this, &writer](mlir::Value val) {
        return writer.getTensor(val);
    };
    const auto inputs = writer.createVector(getInputs() | transformed(getTensorCb));
    const auto outputs = writer.createVector(getOutputs() | transformed(getTensorCb));

    MVCNN::M2ITaskBuilder builder(writer);
    builder.add_src(inputs);
    builder.add_dst(outputs);
    builder.add_do_csc(do_csc());
    builder.add_do_norm(do_norm());
    builder.add_in_fmt(VPUIP::convertM2iColor2MVCNN(inFmt()));
    builder.add_out_fmt(VPUIP::convertM2iColor2MVCNN(outFmt()));

    if (norm().hasValue()) {
        builder.add_norm_coefs(serializedCoefs);
    }

    return {builder.Finish().Union(), MVCNN::SpecificTask_M2ITask};
}
