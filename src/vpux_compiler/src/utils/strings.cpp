//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
