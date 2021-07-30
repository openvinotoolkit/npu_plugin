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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// Dims4D
//

const Dim vpux::IE::Dims4D::Act::N(0);
const Dim vpux::IE::Dims4D::Act::C(1);
const Dim vpux::IE::Dims4D::Act::H(2);
const Dim vpux::IE::Dims4D::Act::W(3);

const Dim vpux::IE::Dims4D::Filter::OC(0);
const Dim vpux::IE::Dims4D::Filter::IC(1);
const Dim vpux::IE::Dims4D::Filter::KY(2);
const Dim vpux::IE::Dims4D::Filter::KX(3);

//
// DataOrderInfo
//

void vpux::IE::DataOrderInfo::printFormat(llvm::raw_ostream& stream) const {
    stream << "Order info [";
    for (size_t i = 0; i < _inputOrders.size(); ++i) {
        stream << " inL[" << i << "]=";
        if (_inputOrders[i].hasValue()) {
            _inputOrders[i]->printFormat(stream);
        } else {
            stream << "ANY";
        }
    }

    stream << ";";
    for (size_t i = 0; i < _outputOrders.size(); ++i) {
        stream << " outL[" << i << "]=";
        if (_outputOrders[i].hasValue()) {
            _outputOrders[i]->printFormat(stream);
        } else {
            stream << "ANY";
        }
    }

    stream << " ]";
}

void vpux::IE::fillDataInfo(IE::DataOrderInfo& info, size_t inNum, size_t outNum, DimsOrder mainOrder) {
    for (size_t i = 0; i < inNum; ++i) {
        info.setInput(i, mainOrder);
    }

    for (size_t i = 0; i < outNum; ++i) {
        info.setOutput(i, mainOrder);
    }
}

//
// LayerOpInterface
//

mlir::LogicalResult vpux::IE::verifyLayer(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyLayer");

    for (auto& arg : op->getOpOperands()) {
        const auto type = arg.get().getType();

        if (type.isa<mlir::MemRefType>()) {
            return errorAt(op, "Layer Operation has MemRef operand #{0}", arg.getOperandNumber());
        }
    }
    for (auto res : op->getOpResults()) {
        const auto type = res.getType();

        if (type.isa<mlir::MemRefType>()) {
            return errorAt(op, "Layer Operation has MemRef result #{0}", res.getResultNumber());
        }
    }

    return mlir::success();
}

IE::DataOrderInfo vpux::IE::getLayerDataOrderInfo(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getLayerDataOrderInfo");

    const auto inputs = op->getOperands();
    const auto outputs = op->getOpResults();

    IE::DataOrderInfo orderInfo{inputs.size(), outputs.size()};

    for (const auto& val : inputs | indexed) {
        orderInfo.setInput(val.index(), DimsOrder::fromValue(val.value()));
    }

    for (const auto& val : outputs | indexed) {
        orderInfo.setOutput(val.index(), DimsOrder::fromValue(val.value()));
    }

    return orderInfo;
}

mlir::LogicalResult vpux::IE::inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                               Optional<mlir::Location> loc, mlir::ValueRange operands,
                                               mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                               SmallVectorImpl<mlir::Type>& inferredTypes) {
    SmallVector<mlir::ShapedTypeComponents> components;
    if (mlir::failed(componentsCb(ctx, loc, operands, attrs, regions, components))) {
        return mlir::failure();
    }

    for (const auto& shapeAndType : components) {
        mlir::Type resType;

        if (shapeAndType.hasRank()) {
            resType = mlir::RankedTensorType::get(shapeAndType.getDims(), shapeAndType.getElementType(),
                                                  shapeAndType.getAttribute());
        } else {
            resType = mlir::UnrankedTensorType::get(shapeAndType.getElementType());
        }

        inferredTypes.push_back(resType);
    }

    return mlir::success();
}

bool vpux::IE::isCompatibleShapeAndElemType(mlir::TypeRange lhs, mlir::TypeRange rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        const auto type1 = std::get<0>(p).dyn_cast<mlir::ShapedType>();
        const auto type2 = std::get<1>(p).dyn_cast<mlir::ShapedType>();

        if (type1 == nullptr || type2 == nullptr) {
            return false;
        }

        if (type1.getShape() != type2.getShape()) {
            return false;
        }

        if (type1.getElementType() != type2.getElementType()) {
            const auto qType1 = type1.getElementType().dyn_cast<mlir::quant::QuantizedType>();
            const auto qType2 = type2.getElementType().dyn_cast<mlir::quant::QuantizedType>();

            if (qType1 == nullptr || qType2 == nullptr) {
                return false;
            }
            if (qType1.getExpressedType() != qType2.getExpressedType()) {
                return false;
            }
            if (qType1.getStorageType() != qType2.getStorageType()) {
                return false;
            }
        }
    }

    return true;
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.cpp.inc>
