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
#include "vpux/compiler/utils/strings.hpp"

#include <fstream>
#include <iomanip>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)  // size_t to integer conversion
#endif

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/GraphWriter.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace vpux {
namespace VPU {

Json readManualStrategyJSON(StringRef fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for input strategy json was not provided");

    std::ifstream i(fileName.data());

    Json json;
    if (i.good()) {
        i >> json;
    }

    return json;
}

void writeManualStrategyJSON(StringRef fileName, Json& json) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for output strategy json was not provided");

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
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
}

mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, Json& newAttrVal) {
    if (oldAttr.isa<mlir::StringAttr>()) {
        // cast to std::string so it can be compared with std::string
        return mlir::StringAttr::get(oldAttr.getContext(), static_cast<std::string>(newAttrVal.begin().value()));
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

Json createStrategyJSONFromOperations(Json& json, llvm::DenseMap<mlir::Location, mlir::Operation*>& operations,
                                      ArrayRef<StringRef> strategyAttributes) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        auto parentClusterOp = op.second->getParentOfType<VPU::NCEClusterTilingOp>();
        // retrieve related attributes and save in JSON
        for (auto attribute : strategyAttributes) {
            Json strategyAsJSON;
            if (op.second->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToString(op.second->getAttr(attribute));
            } else if (parentClusterOp != nullptr && parentClusterOp->hasAttr(attribute)) {
                strategyAsJSON = convertAttrToString(parentClusterOp->getAttr(attribute));
            } else if (json.contains(opName) && json[opName].contains(attribute.data())) {
                strategyAsJSON = json[opName].at(attribute.data());
            } else {
                // currently no default strategy, set NONE
                strategyAsJSON = "NONE";
            }
            json[opName][attribute.data()] = strategyAsJSON;
        }
    }

    return json;
}

void overwriteManualStrategy(Json& manualStrategy, llvm::DenseMap<mlir::Location, mlir::Operation*>& operations) {
    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        // check if manual strategy for layer exists
        if (!manualStrategy.contains(opName)) {
            continue;
        }
        auto currOpStrategy = manualStrategy.at(opName);
        for (Json::iterator it = currOpStrategy.begin(); it != currOpStrategy.end(); ++it) {
            // replace attributes of the operation (skip NONE) using it.value()
            if (it.value() != "NONE") {
                mlir::Attribute manualAttribute;
                if (it.key() == "multiClusterStrategy") {
                    auto dummyAttr = mlir::StringAttr::get(op.second->getContext(), "");
                    manualAttribute = convertJSONToAttr(dummyAttr, it.value());
                } else if (it.key() == "tilingStrategy") {
                    // tiling case, where strategy selection and IR modification occurs in a single pass
                    // TODO: remove "else" when tiling strategy will be abstracted into strategy pass
                    auto dummyAttr = getIntArrayAttr(op.second->getContext(), Shape(4));
                    manualAttribute = convertJSONToAttr(dummyAttr, it.value());
                } else {
                    VPUX_THROW("Unsupported attribute '{0}'", it.key());
                }
                op.second->setAttr(it.key(), manualAttribute);
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
