//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/core/profiling.hpp"

#include <iostream>

using namespace vpux;

std::string vpux::stringifyLocation(mlir::Location location) {
    std::ostringstream ostr;

    unsigned index = 0;
    //
    // Walk the location structure and collect string names. This will allow
    // to analyze both simple NameLoc as well as compound FusedLoc
    //
    location->walk([&](mlir::Location loc) {
        if (const auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
            if (index > 0) {
                const auto separator = (index == 1) ? LOCATION_ORIGIN_SEPARATOR : LOCATION_SEPARATOR;
                ostr << separator;
            }
            ostr << nameLoc.getName().strref().data();
            index++;
        }

        return mlir::WalkResult::advance();
    });

    return ostr.str();
}
