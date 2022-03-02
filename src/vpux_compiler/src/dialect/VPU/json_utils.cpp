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

std::string convertAttrToString(mlir::Attribute attr) {
    if (attr.isa<mlir::StringAttr>()) {
        return attr.cast<mlir::StringAttr>().getValue().data();
    } else {
        VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
        return "implement conversion";
    }
}

mlir::Attribute convertStringToAttr(mlir::Attribute oldAttr, std::string newAttrVal) {
    if (oldAttr.isa<mlir::StringAttr>()) {
        return mlir::StringAttr::get(oldAttr.getContext(), newAttrVal);
    } else {
        VPUX_THROW("Conversion from string to this attribute '{0}' not implemented", oldAttr);
        return nullptr;
    }
}

Json createStrategyJSONFromOperations(llvm::DenseMap<mlir::Location, mlir::Operation*> operations,
                                      SmallVector<StringRef> strategyAttributes) {
    Json j;

    for (auto& op : operations) {
        const auto opName = vpux::stringifyLocation(op.first);
        // retrieve related attributes and save in JSON
        for (auto attribute : strategyAttributes) {
            if (op.second->hasAttr(attribute)) {
                auto stringAttr = convertAttrToString(op.second->getAttr(attribute));
                j[opName][attribute.data()] = stringAttr;
            }
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
                // replace attributes of the operation using it.value()
                if (op.second->hasAttr(it.key())) {
                    auto newAttr = convertStringToAttr(op.second->getAttr(it.key()), it.value());
                    op.second->setAttr(it.key(), newAttr);
                }
            }
        }
    }

    return;
}

}  // namespace VPU
}  // namespace vpux
