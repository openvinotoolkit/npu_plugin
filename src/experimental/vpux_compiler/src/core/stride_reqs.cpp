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

#include "vpux/compiler/core/stride_reqs.hpp"

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

StringRef vpux::EnumTraits<StrideReqKind>::getEnumClassName() {
    return "StrideReqKind";
}

StringRef vpux::EnumTraits<StrideReqKind>::getEnumValueName(StrideReqKind val) {
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

Optional<StrideReqKind> vpux::EnumTraits<StrideReqKind>::parseEnumValue(StringRef valStr) {
    if (valStr == "Compact") {
        return StrideReqKind::Compact;
    }
    if (valStr == "Aligned") {
        return StrideReqKind::Aligned;
    }
    if (valStr == "Fixed") {
        return StrideReqKind::Fixed;
    }
    return None;
}

//
// DimStrideReq
//

void vpux::DimStrideReq::verifyAttrs(StrideReqKind kind, int64_t extraValue) {
    if (kind == StrideReqKind::Compact) {
        VPUX_THROW_UNLESS(extraValue == 0, "Got non zero extraValue {0} for {1} DimStrideReq", extraValue, kind);

        return;
    }

    VPUX_THROW_UNLESS(extraValue > 0, "Got negative or zero extraValue {0} in {1} DimStrideReq", extraValue, kind);

    if (kind == StrideReqKind::Aligned) {
        VPUX_THROW_UNLESS(isPowerOfTwo(extraValue), "Alignment value {0} is not a power-of-two in {1} DimStrideReq",
                          extraValue, kind);
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

StrideReqs vpux::StrideReqs::simple() {
    StrideReqs r;
    r.add(DimStrideReq::compact(MemDim(0)));
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

void vpux::StrideReqs::calcStrides(MemStrides& memStrides, int64_t elemByteSize, MemShapeRef memShape) const {
    StrideReqsRef(*this).calcStrides(memStrides, elemByteSize, memShape);
}

MemStrides vpux::StrideReqs::calcStrides(int64_t elemByteSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).calcStrides(elemByteSize, memShape);
}

bool vpux::StrideReqs::checkStrides(MemStridesRef memStrides, int64_t elemByteSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).checkStrides(memStrides, elemByteSize, memShape);
}

StrideReqs vpux::StrideReqs::join(StrideReqsRef other, int64_t elemByteSize, MemShapeRef memShape) const {
    return StrideReqsRef(*this).join(other, elemByteSize, memShape);
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

int64_t applyStrideReq(int64_t baseStride, const DimStrideReq& req) {
    if (req.kind() == StrideReqKind::Compact) {
        return baseStride;
    } else if (req.kind() == StrideReqKind::Aligned) {
        return alignVal(baseStride, req.alignment());
    } else if (req.kind() == StrideReqKind::Fixed) {
        return req.fixedValue();
    } else {
        VPUX_THROW("Uncovered stride requirement {0}", req);
    }
}

}  // namespace

void vpux::StrideReqsRef::calcStrides(MemStrides& memStrides, int64_t elemByteSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());

    memStrides.resize(memShape.size());

    if (memShape.empty()) {
        return;
    }

    for (const auto ind : irange(memShape.size())) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];

        if (ind == 0) {
            memStrides[memDim] = elemByteSize;
        } else {
            const auto prevMemDim = MemDim(ind - 1);
            memStrides[memDim] = memStrides[prevMemDim] * memShape[prevMemDim];
        }

        if (req.hasValue()) {
            memStrides[memDim] = applyStrideReq(memStrides[memDim], req.getValue());
        }
    }
}

MemStrides vpux::StrideReqsRef::calcStrides(int64_t elemByteSize, MemShapeRef memShape) const {
    MemStrides memStrides;
    calcStrides(memStrides, elemByteSize, memShape);
    return memStrides;
}

bool vpux::StrideReqsRef::checkStrides(MemStridesRef memStrides, int64_t elemByteSize, MemShapeRef memShape) const {
    assert(memShape.isStatic());
    assert(memStrides.isStatic());
    assert(memStrides.size() == memShape.size());

    for (const auto ind : irange(memShape.size())) {
        const auto memDim = MemDim(ind);
        const auto req = (*this)[memDim];
        const auto strideVal = memStrides[memDim];

        if (ind == 0) {
            if (strideVal < elemByteSize) {
                return false;
            }
        } else {
            const auto prevMemDim = MemDim(ind - 1);

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
            if (ind == 0) {
                if (strideVal != elemByteSize) {
                    return false;
                }
            } else {
                const auto prevMemDim = MemDim(ind - 1);

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

StrideReqs vpux::StrideReqsRef::join(StrideReqsRef other, int64_t elemByteSize, MemShapeRef memShape) const {
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

    const auto mergedStrides = merged.calcStrides(elemByteSize, memShape);

    VPUX_THROW_UNLESS(checkStrides(mergedStrides, elemByteSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemByteSize {2}, memShape {3})",
                      mergedStrides, *this, elemByteSize, memShape);
    VPUX_THROW_UNLESS(other.checkStrides(mergedStrides, elemByteSize, memShape),
                      "StrideReqs::join : mergedStrides {0} doesn't satisfy original "
                      "StrideReqs {1} (elemByteSize {2}, memShape {3})",
                      mergedStrides, other, elemByteSize, memShape);

    return merged;
}

void vpux::StrideReqsRef::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "{0}", raw());
}

StrideReqs vpux::StrideReqsRef::toValues() const {
    return StrideReqs(StrideReqs::ContainerType(begin(), end()));
}
