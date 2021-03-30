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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

using namespace vpux;

//
// StrideReqKind
//

StringLiteral vpux::stringifyEnum(StrideReqKind val) {
    switch (val) {
    case StrideReqKind::Compact:
        return "Compact";
    case StrideReqKind::Aligned:
        return "Aligned";
    case StrideReqKind::Fixed:
        return "Fixed";
    default:
        return "<UNKNOWN>";
    }
}

//
// DimStrideReq
//

void vpux::DimStrideReq::verifyAttrs(StrideReqKind kind, Bit extraValue) {
    if (kind == StrideReqKind::Compact) {
        VPUX_THROW_UNLESS(extraValue.count() == 0, "Got non zero extraValue {0} for {1} DimStrideReq", extraValue,
                          kind);

        return;
    }

    VPUX_THROW_UNLESS(extraValue.count() > 0, "Got negative or zero extraValue {0} in {1} DimStrideReq", extraValue,
                      kind);

    if (kind == StrideReqKind::Aligned) {
        VPUX_THROW_UNLESS(isPowerOfTwo(extraValue.count()),
                          "Alignment value {0} is not a power-of-two in {1} DimStrideReq", extraValue, kind);
    }
}

void vpux::DimStrideReq::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}:{1}", memDim(), kind());

    switch (kind()) {
    case StrideReqKind::Aligned:
    case StrideReqKind::Fixed:
        printTo(stream, ":{0}", _extraValue);
        break;

    default:
        break;
    }
}

bool vpux::operator==(const DimStrideReq& req1, const DimStrideReq& req2) {
    return req1.memDim() == req2.memDim() && req1.kind() == req2.kind() && req1.extraValue() == req2.extraValue();
}

bool vpux::operator!=(const DimStrideReq& req1, const DimStrideReq& req2) {
    return !(req1 == req2);
}

//
// StrideReqs
//

StrideReqs vpux::StrideReqs::simple(size_t numDims) {
    numDims = numDims ? numDims - 1 : 0;

    StrideReqs r;
    r.add(DimStrideReq::compact(MemDim(numDims)));
    return r;
}

StrideReqs vpux::StrideReqs::compact(size_t numDims) {
    DimsOrder::validateNumDims(numDims);

    StrideReqs r;
    for (size_t d = 0; d < numDims; ++d) {
        r.add(DimStrideReq::compact(MemDim(d)));
    }
    return r;
}

StrideReqs vpux::StrideReqs::fixed(StridesRef strides) {
    StrideReqs r;
    for (const auto& p : strides | indexed) {
        r.add(DimStrideReq::fixed(MemDim(p.index()), p.value()));
    }
    return r;
}

StrideReqs& vpux::StrideReqs::add(const DimStrideReq& req) {
    assert(!hasReqFor(req.memDim()));

    _cont.push_back(req);
    return *this;
}

StrideReqs& vpux::StrideReqs::remove(MemDim memDim) {
    const auto it = std::remove_if(_cont.begin(), _cont.end(), [memDim](const DimStrideReq& req) {
        return req.memDim() == memDim;
    });
    _cont.erase(it, _cont.end());
    return *this;
}

bool vpux::StrideReqs::hasReqFor(MemDim memDim) const {
    return (*this)[memDim].hasValue();
}

Optional<DimStrideReq> vpux::StrideReqs::operator[](MemDim memDim) const {
    return StrideReqsRef(*this)[memDim];
}

void vpux::StrideReqs::calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const {
    StrideReqsRef(*this).calcStrides(memStrides, elemSize, memShape);
}

MemStrides vpux::StrideReqs::calcStrides(Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).calcStrides(elemSize, memShape);
}

MemStrides vpux::StrideReqs::calcStrides(DimsOrder order, mlir::ShapedType type) const {
    return StrideReqsRef(*this).calcStrides(order, type);
}

bool vpux::StrideReqs::checkStrides(mlir::MemRefType type) const {
    return StrideReqsRef(*this).checkStrides(type);
}

bool vpux::StrideReqs::checkStrides(mlir::Value val) const {
    return StrideReqsRef(*this).checkStrides(val);
}

bool vpux::StrideReqs::checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).checkStrides(memStrides, elemSize, memShape);
}

StrideReqs vpux::StrideReqs::join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).join(other, elemSize, memShape);
}

void vpux::StrideReqs::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", raw());
}

//
// StrideReqsRef
//

bool vpux::StrideReqsRef::hasReqFor(MemDim memDim) const {
    return (*this)[memDim].hasValue();
}

Optional<DimStrideReq> vpux::StrideReqsRef::operator[](MemDim memDim) const {
    const auto it = std::find_if(_ref.begin(), _ref.end(), [memDim](const DimStrideReq& req) {
        return req.memDim() == memDim;
    });

    if (it != _ref.end()) {
        return *it;
    } else {
        return None;
    }
}

namespace {

Bit applyStrideReq(Bit baseStride, const DimStrideReq& req) {
    if (req.kind() == StrideReqKind::Compact) {
        return baseStride;
    } else if (req.kind() == StrideReqKind::Aligned) {
        return Bit(alignVal(baseStride.count(), req.alignment().count()));
    } else if (req.kind() == StrideReqKind::Fixed) {
        return req.fixedValue();
    } else {
        VPUX_THROW("Uncovered stride requirement {0}", req);
    }
}

}  // namespace

void vpux::StrideReqsRef::calcStrides(MemStrides& memStrides, Bit elemSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());

    memStrides.resize(memShape.size());

    if (memShape.empty()) {
        return;
    }

    for (const auto ind : irange(memShape.size()) | reversed) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];

        if (ind == memShape.size() - 1) {
            memStrides[memDim] = elemSize;
        } else {
            const auto prevMemDim = MemDim(ind + 1);
            memStrides[memDim] = memStrides[prevMemDim] * memShape[prevMemDim];
        }

        if (req.hasValue()) {
            memStrides[memDim] = applyStrideReq(memStrides[memDim], req.getValue());
        }
    }
}

MemStrides vpux::StrideReqsRef::calcStrides(Bit elemSize, MemShapeRef memShape) const {
    MemStrides memStrides;
    calcStrides(memStrides, elemSize, memShape);
    return memStrides;
}

MemStrides vpux::StrideReqsRef::calcStrides(DimsOrder order, mlir::ShapedType type) const {
    const auto elemSize = getElemTypeSize(type);
    const auto shape = getShape(type);
    const auto memShape = order.toMemoryOrder(shape);
    return calcStrides(elemSize, memShape);
}

bool vpux::StrideReqsRef::checkStrides(mlir::MemRefType type) const {
    const auto elemSize = getElemTypeSize(type);
    const auto shape = getShape(type);
    const auto strides = getStrides(type);
    const auto order = DimsOrder::fromType(type);
    const auto memShape = order.toMemoryOrder(shape);
    const auto memStrides = order.toMemoryOrder(strides);
    return checkStrides(memStrides, elemSize, memShape);
}

bool vpux::StrideReqsRef::checkStrides(mlir::Value val) const {
    const auto type = val.getType().dyn_cast_or_null<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non MemRefType '{1}'", val, val.getType());
    return checkStrides(type);
}

bool vpux::StrideReqsRef::checkStrides(MemStridesRef memStrides, Bit elemSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());
    assert(memStrides.isStatic());
    assert(memStrides.size() == memShape.size());

    for (const auto ind : irange(memShape.size())) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];
        const auto strideVal = memStrides[memDim];

        if (ind == memShape.size() - 1) {
            if (strideVal < elemSize) {
                return false;
            }
        } else {
            const auto prevMemDim = MemDim(ind + 1);

            if (strideVal < memStrides[prevMemDim] * memShape[prevMemDim]) {
                return false;
            }
        }

        if (!req.hasValue()) {
            continue;
        } else if (req.getValue().kind() == StrideReqKind::Fixed) {
            if (strideVal != req.getValue().fixedValue()) {
                return false;
            }
        } else if (req.getValue().kind() == StrideReqKind::Aligned) {
            if (strideVal % req.getValue().alignment() != 0) {
                return false;
            }
        } else if (req.getValue().kind() == StrideReqKind::Compact) {
            if (ind == memShape.size() - 1) {
                if (strideVal != elemSize) {
                    return false;
                }
            } else {
                const auto prevMemDim = MemDim(ind + 1);

                if (strideVal != memStrides[prevMemDim] * memShape[prevMemDim]) {
                    return false;
                }
            }
        } else {
            VPUX_THROW("Uncovered stride requirement {0}", req);
        }
    }

    return true;
}

StrideReqs vpux::StrideReqsRef::join(StrideReqsRef other, Bit elemSize, MemShapeRef memShape) const {
    StrideReqs merged;

    const auto mergeReq = [&merged](const DimStrideReq& curReq, const DimStrideReq& otherReq) {
        assert(curReq.memDim() == otherReq.memDim());

        if (curReq.kind() == StrideReqKind::Fixed && otherReq.kind() == StrideReqKind::Fixed) {
            VPUX_THROW_UNLESS(curReq.fixedValue() == otherReq.fixedValue(),
                              "StrideReqs::join : got fixed strides with "
                              "different values at MemDim {0} : "
                              "{1} and {2}",
                              curReq.memDim(), curReq.fixedValue(), otherReq.fixedValue());
        }

        if (otherReq.kind() == StrideReqKind::Fixed) {
            merged.add(otherReq);
        } else {
            merged.add(curReq);
        }
    };

    for (const auto& curReq : *this) {
        const auto otherReq = other[curReq.memDim()];

        if (otherReq.hasValue()) {
            mergeReq(curReq, otherReq.getValue());
        } else {
            merged.add(curReq);
        }
    }

    for (const auto& otherReq : other) {
        const auto curReq = (*this)[otherReq.memDim()];

        if (curReq.hasValue()) {
            mergeReq(curReq.getValue(), otherReq);
        } else {
            merged.add(otherReq);
        }
    }

    const auto mergedStrides = merged.calcStrides(elemSize, memShape);

    VPUX_THROW_UNLESS(checkStrides(mergedStrides, elemSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemSize {2}, memShape {3})",
                      mergedStrides, *this, elemSize, memShape);
    VPUX_THROW_UNLESS(other.checkStrides(mergedStrides, elemSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemSize {2}, memShape {3})",
                      mergedStrides, other, elemSize, memShape);

    return merged;
}

void vpux::StrideReqsRef::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", raw());
}

StrideReqs vpux::StrideReqsRef::toValues() const {
    return StrideReqs(StrideReqs::ContainerType(begin(), end()));
}
