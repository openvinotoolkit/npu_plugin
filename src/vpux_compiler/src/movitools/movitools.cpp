#include "vpux/compiler/movitools/movitools.h"
#include <mlir/Support/DebugStringHelper.h>
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>

using namespace llvm;  // NOLINT

namespace vpux {
namespace movitools {

std::string getMoviToolsDir() {
    const auto toolsDir = llvm::sys::Process::GetEnv("MV_TOOLS_DIR");
    VPUX_THROW_UNLESS(toolsDir.hasValue(), "MV_TOOLS_DIR env var must be set");

    const auto toolsVer = llvm::sys::Process::GetEnv("MV_TOOLS_VERSION");
    VPUX_THROW_UNLESS(toolsVer.hasValue(), "MV_TOOLS_VERSION env var must be set");

    SmallString<128> finalToolsDir(toolsDir.getValue());
    sys::path::append(finalToolsDir, toolsVer.getValue());
    VPUX_THROW_UNLESS(sys::fs::is_directory(finalToolsDir), "{0} is not a directory", finalToolsDir.str());

    return finalToolsDir.str().str();
}

}  // namespace movitools
}  // namespace vpux
