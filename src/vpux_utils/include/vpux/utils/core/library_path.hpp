#pragma once

#include <file_utils.h>

namespace vpux {

static std::string getLibFilePath(const std::string& baseName) {
    return FileUtils::makePluginLibraryName(InferenceEngine::getIELibraryPath(), baseName + IE_BUILD_POSTFIX);
}

}  // namespace vpux
