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

#pragma once

#include <cassert>
#include <unordered_map>
#include <unordered_set>

namespace vpux {

struct ScheduledOpOneResource {
    using OperationType = size_t;
    ScheduledOpOneResource(): _op(), _addressStart(), _addressEnd() {
    }

    ScheduledOpOneResource(OperationType op, size_t start, size_t end)
            : _op(op), _addressStart(start), _addressEnd(end) {
    }

    ScheduledOpOneResource(const ScheduledOpOneResource& o)
            : _op(o._op), _addressStart(o._addressStart), _addressEnd(o._addressEnd) {
    }

    const ScheduledOpOneResource& operator=(const ScheduledOpOneResource& o) {
        _op = o._op;
        _addressStart = o._addressStart;
        _addressEnd = o._addressEnd;
        return *this;
    }

    bool operator==(const ScheduledOpOneResource& o) const {
        return (_op == o._op) && (_addressStart == o._addressStart) && (_addressEnd == o._addressEnd);
    }

    OperationType _op;
    size_t _addressStart;
    size_t _addressEnd;
};  // struct ScheduledOpOneResource //

}  // namespace vpux
