#pragma once

#include <openvino/util/file_util.hpp>

namespace vpux {

static std::string getLibFilePath(const std::string& baseName) {
    return ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
}

}  // namespace vpux
