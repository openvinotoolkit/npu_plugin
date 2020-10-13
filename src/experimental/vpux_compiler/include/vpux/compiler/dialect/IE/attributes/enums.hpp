//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/core/dims_order.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/mlir/attributes.hpp"

#include <mlir/IR/AffineMap.h>

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/enums.hpp.inc>

//
// EnumTraits
//

namespace vpux {

template <>
struct EnumTraits<IE::Layout> final {
    static auto getEnumValueName(IE::Layout val) {
        return IE::stringifyEnum(val);
    }

    static auto parseValue(StringRef valStr) {
        return IE::symbolizeEnum<IE::Layout>(valStr);
    }
};

}  // namespace vpux

//
// LayoutAttr
//

namespace vpux {
namespace IE {

class LayoutAttr final : public EnumAttrBase<LayoutAttr, Layout> {
public:
    using EnumAttrBase<LayoutAttr, Layout>::EnumAttrBase;

public:
    static StringRef getMnemonic();

public:
    static int32_t getRank(Layout layout);
    int32_t getRank() const;

    static DimsOrder toDimsOrder(Layout layout);
    DimsOrder toDimsOrder() const;

    static mlir::AffineMap toAffineMap(mlir::MLIRContext* ctx, Layout layout);
    mlir::AffineMap toAffineMap() const;
};

}  // namespace IE
}  // namespace vpux
