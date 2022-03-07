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

#include <fstream>
#include <iomanip>
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/json.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/strings.hpp"

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

Json readManualStrategyJSON(StringRef fileName);
void writeManualStrategyJSON(StringRef fileName, Json json);

Json convertAttrToString(mlir::Attribute attr);
Json createStrategyJSONFromOperations(Json j, llvm::DenseMap<mlir::Location, mlir::Operation*> operations,
                                      SmallVector<StringRef> strategyAttributes);
mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, Json newAttrVal);
void overwriteManualStrategy(Json manualStrategy, llvm::DenseMap<mlir::Location, mlir::Operation*> operations);

}  // namespace VPU
}  // namespace vpux
