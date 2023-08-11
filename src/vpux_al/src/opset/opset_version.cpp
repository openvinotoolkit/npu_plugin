//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/al/opset/opset_version.hpp"

//
// tryStrToInt
//
uint32_t vpux::tryStrToInt(const std::string& strVersion) {
    try {
        const uint32_t intVersion = std::stoi(strVersion);
        return intVersion;
    } catch (std::invalid_argument const& ex) {
        VPUX_THROW("During tryStrToInt, std::invalid_argument::what(): '{0}'", ex.what());
    } catch (std::out_of_range const& ex) {
        VPUX_THROW("During tryStrToInt, std::out_of_range::what(): '{0}'", ex.what());
    }
}

//
// extractOpsetVersion
//
uint32_t vpux::extractOpsetVersion() {
    uint32_t largestVersion = 0;
    const auto& availOpsets = ov::get_available_opsets();
    const std::string opset = "opset";
    for (auto const& kv : availOpsets) {
        const std::string& opsetCurrent = kv.first;
        const std::size_t found = opsetCurrent.find(opset);
        const std::string strVersion = opsetCurrent.substr(found + opset.length());
        const uint32_t intVersion = tryStrToInt(strVersion);
        if (intVersion > largestVersion) {
            largestVersion = intVersion;
        }
    }
    return largestVersion;
}
