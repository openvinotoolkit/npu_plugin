//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <fstream>
#include <iomanip>

namespace vpux {
namespace VPU {

Json readManualStrategyJSON(StringRef fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for input strategy json was not provided");

    std::ifstream i(fileName.str());
    VPUX_THROW_UNLESS(i.good(), "File with manual strategy not opened correctly");

    Json json;
    i >> json;

    return json;
}

void writeManualStrategyJSON(StringRef fileName, const Json& json) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for output strategy json was not provided");

    std::ofstream o(fileName.str());
    VPUX_THROW_UNLESS(o.good(), "File with manual strategy not created correctly");
    o << std::setw(4) << json << std::endl;

    return;
}

Json convertAttrToJSON(mlir::Attribute attr) {
    if (attr.isa<mlir::StringAttr>()) {
        Json clusteringStrategy = attr.cast<mlir::StringAttr>().getValue().str();
        return clusteringStrategy;
    } else if (attr.isa<mlir::ArrayAttr>()) {
        auto values = Shape(parseIntArrayAttr<int64_t>(attr.cast<mlir::ArrayAttr>()));
        VPUX_THROW_UNLESS(values.size() == 4, "Shape has fewer dimensions than expected (4), got '{0}'", values.size());
        Json tilingStrategy;
        tilingStrategy["N"] = values[Dims4D::Act::N];
        tilingStrategy["C"] = values[Dims4D::Act::C];
        tilingStrategy["H"] = values[Dims4D::Act::H];
        tilingStrategy["W"] = values[Dims4D::Act::W];

        return tilingStrategy;
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
}

mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, const Json& newAttrVal) {
    if (oldAttr.isa<mlir::StringAttr>()) {
        // cast to std::string so it can be compared with std::string
        return mlir::StringAttr::get(oldAttr.getContext(), newAttrVal.get<std::string>());
    } else if (oldAttr.isa<mlir::ArrayAttr>()) {
        Shape newShape(4, 1);
        newShape[Dims4D::Act::N] = static_cast<int64_t>(std::stoi(newAttrVal.at("N").begin().value().dump()));
        newShape[Dims4D::Act::C] = static_cast<int64_t>(std::stoi(newAttrVal.at("C").begin().value().dump()));
        newShape[Dims4D::Act::H] = static_cast<int64_t>(std::stoi(newAttrVal.at("H").begin().value().dump()));
        newShape[Dims4D::Act::W] = static_cast<int64_t>(std::stoi(newAttrVal.at("W").begin().value().dump()));
        return getIntArrayAttr(oldAttr.getContext(), newShape);
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", oldAttr);
}

void createStrategyJSONFromOperations(Json& json, llvm::MapVector<mlir::Location, mlir::Operation*>& operations,
                                      ArrayRef<StringRef> strategyAttributes) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        auto parentClusterOp = op.second->getParentOfType<VPU::NCEClusterTilingOp>();
        // retrieve related attributes and save in JSON
        for (auto attribute : strategyAttributes) {
            Json strategyAsJSON;
            if (op.second->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToJSON(op.second->getAttr(attribute));
            } else if (parentClusterOp != nullptr && parentClusterOp->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToJSON(parentClusterOp->getAttr(attribute));
            } else if (json.contains(opName) && json[opName].contains(attribute.str())) {
                strategyAsJSON = json[opName].at(attribute.str());
            } else {
                // currently no default strategy, set NONE
                strategyAsJSON = defaultNoStrategy.str();
            }
            json[opName][attribute.str()] = strategyAsJSON;
        }
    }
}

void overwriteManualStrategy(Json& manualStrategy, llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        // check if manual strategy for layer exists
        if (!manualStrategy.contains(opName)) {
            continue;
        }
        auto currOpStrategy = manualStrategy.at(opName);
        for (Json::iterator it = currOpStrategy.begin(); it != currOpStrategy.end(); ++it) {
            // replace attributes of the operation (skip NONE) using it.value()
            if (it.value() != defaultNoStrategy.str()) {
                if (it.key() == multiClusterStrategy) {
                    auto dummyAttr = mlir::StringAttr::get(op.second->getContext(), "");
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it.value());
                    if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op.second)) {
                        clusteredOp.setMultiClusterStrategyAttr(
                                manualAttribute.cast<vpux::VPU::MultiClusterStrategyAttr>().getValue());
                    }
                } else if (it.key() == tilingStrategy) {
                    // tiling case, where strategy selection and IR modification occurs in a single pass
                    // TODO: remove "else" when tiling strategy will be abstracted into strategy pass
                    auto dummyAttr = getIntArrayAttr(op.second->getContext(), Shape(4));
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it.value());
                    op.second->setAttr(manualTilingStrategy, manualAttribute);
                } else {
                    VPUX_THROW("Unsupported attribute '{0}'", it.key());
                }
            } else {
                if (op.second->hasAttr(it.key())) {
                    // currently no default value, to disable multiclustering remove the attribute
                    op.second->removeAttr(it.key());
                }
            }
        }
    }
}

}  // namespace VPU
}  // namespace vpux
