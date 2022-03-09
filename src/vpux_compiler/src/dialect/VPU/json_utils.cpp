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

#include "vpux/compiler/dialect/VPU/json_utils.hpp"

namespace vpux {
namespace VPU {

Json readManualStrategyJSON(StringRef fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for PrintDot was not provided");

    std::ifstream i(fileName.data());

    Json j;
    if (i.good()) {
        i >> j;
    }

    return j;
}

void writeManualStrategyJSON(StringRef fileName, Json json) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for PrintDot was not provided");

    std::ofstream o(fileName.data());
    o << std::setw(4) << json << std::endl;

    return;
}

Json convertAttrToString(mlir::Attribute attr) {
    if (attr.isa<mlir::StringAttr>()) {
        Json clusteringStrategy;
        clusteringStrategy = attr.cast<mlir::StringAttr>().getValue().data();
        return clusteringStrategy;
    } else if (attr.isa<mlir::ArrayAttr>()) {
        auto values = Shape(parseIntArrayAttr<int64_t>(attr.cast<mlir::ArrayAttr>()));

        Json tilingStrategy;
        tilingStrategy["N"] = values[Dims4D::Act::N];
        tilingStrategy["C"] = values[Dims4D::Act::C];
        tilingStrategy["H"] = values[Dims4D::Act::H];
        tilingStrategy["W"] = values[Dims4D::Act::W];

        return tilingStrategy;
    } else {
        VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
        return nullptr;
    }
}

mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, Json newAttrVal) {
    if (oldAttr.isa<mlir::StringAttr>()) {
        // cast to std::string so it can be compared with std::string
        return mlir::StringAttr::get(oldAttr.getContext(), (std::string)newAttrVal.begin().value());
    } else if (oldAttr.isa<mlir::ArrayAttr>()) {
        Shape newShape(4, 1);
        newShape[Dims4D::Act::N] = static_cast<int64_t>(std::stoi(newAttrVal.at("N").begin().value().dump()));
        newShape[Dims4D::Act::C] = static_cast<int64_t>(std::stoi(newAttrVal.at("C").begin().value().dump()));
        newShape[Dims4D::Act::H] = static_cast<int64_t>(std::stoi(newAttrVal.at("H").begin().value().dump()));
        newShape[Dims4D::Act::W] = static_cast<int64_t>(std::stoi(newAttrVal.at("W").begin().value().dump()));
        return getIntArrayAttr(oldAttr.getContext(), newShape);
    } else {
        VPUX_THROW("Conversion from string to this attribute '{0}' not implemented", oldAttr);
        return nullptr;
    }
}

Json createStrategyJSONFromOperations(Json j, llvm::DenseMap<mlir::Location, mlir::Operation*> operations,
                                      SmallVector<StringRef> strategyAttributes) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        auto paretClusterOp = op.second->getParentOfType<VPU::NCEClusterTilingOp>();
        // retrieve related attributes and save in JSON
        for (auto attribute : strategyAttributes) {
            Json strategyAsJSON;
            if (op.second->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToString(op.second->getAttr(attribute));
            } else if (paretClusterOp != nullptr && paretClusterOp->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToString(paretClusterOp->getAttr(attribute));
            } else if (j.find(opName) != j.end() && j[opName].find(attribute.data()) != j[opName].end()) {
                strategyAsJSON = j[opName].at(attribute.data());
            } else {
                // currently no default strategy, set NONE
                strategyAsJSON = "NONE";
            }
            j[opName][attribute.data()] = strategyAsJSON;
        }
    }

    return j;
}

void overwriteManualStrategy(Json manualStrategy, llvm::DenseMap<mlir::Location, mlir::Operation*> operations) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        // check if manual strategy for layer exists
        if (manualStrategy.find(opName) != manualStrategy.end()) {
            auto currOpStrategy = manualStrategy.at(opName);
            for (Json::iterator it = currOpStrategy.begin(); it != currOpStrategy.end(); ++it) {
                // replace attributes of the operation (skip NONE) using it.value()
                if (it.value() != "NONE") {
                    mlir::Attribute manualAttribute;
                    if (op.second->hasAttr(it.key())) {
                        manualAttribute = convertJSONToAttr(op.second->getAttr(it.key()), it.value());
                    } else {
                        // tiling case, where strategy selection and IR modification occurs in a single pass
                        // TODO: remove "else" when tiling strategy will be abstracted into strategy pass
                        auto dummyAttr = getIntArrayAttr(op.second->getContext(), Shape(4));
                        manualAttribute = convertJSONToAttr(dummyAttr, it.value());
                    }
                    op.second->setAttr(it.key(), manualAttribute);
                } else {
                    if (op.second->hasAttr(it.key())) {
                        // currently no default value, to diable multiclustering remove the attribute
                        op.second->removeAttr(it.key());
                    }
                }
            }
        }
    }
}

}  // namespace VPU
}  // namespace vpux
