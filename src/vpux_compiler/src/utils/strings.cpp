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

#include "vpux/compiler/utils/strings.hpp"

#include <iostream>

using namespace vpux;

std::string vpux::stringifyLocation(mlir::Location location) {
    std::ostringstream ostr;

    bool isFirstItem = true;
    //
    // Walk the location structure and collect string names. This will allow
    // to analyze both simple NameLoc as well as compound FusedLoc
    //
    location->walk([&](mlir::Location loc) {
        if (const auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
            if (!isFirstItem) {
                ostr << "/";
            }

            ostr << nameLoc.getName().strref().data();
            isFirstItem = false;
        }

        return mlir::WalkResult::advance();
    });

    return ostr.str();
}
