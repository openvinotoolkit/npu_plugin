//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cassert>
#include <unordered_map>
#include <unordered_set>

namespace vpux {

struct ScheduledOpOneResource {
    enum class EResRelation { PRODUCER = 0, CONSUMER = 1 };

    using OperationType = size_t;
    ScheduledOpOneResource() = default;

    ScheduledOpOneResource(OperationType op, size_t start, size_t end,
                           EResRelation resRelation = EResRelation::PRODUCER)
            : _op(op), _addressStart(start), _addressEnd(end), _resRelation(resRelation) {
    }

    ScheduledOpOneResource(const ScheduledOpOneResource& o)
            : _op(o._op), _addressStart(o._addressStart), _addressEnd(o._addressEnd), _resRelation(o._resRelation) {
    }

    ScheduledOpOneResource& operator=(const ScheduledOpOneResource& o) {
        if (&o == this) {
            return *this;
        }

        _op = o._op;
        _addressStart = o._addressStart;
        _addressEnd = o._addressEnd;
        return *this;
    }

    bool operator==(const ScheduledOpOneResource& o) const {
        return (_op == o._op) && (_addressStart == o._addressStart) && (_addressEnd == o._addressEnd);
    }

    bool operator<(const ScheduledOpOneResource& o) const {
        if (_addressStart == o._addressStart && _addressEnd == o._addressEnd) {
            return _op < o._op;
        }
        return (_addressStart != o._addressStart) ? (_addressStart < o._addressStart) : (_addressEnd < o._addressEnd);
    }
    ~ScheduledOpOneResource() = default;

    OperationType _op{};
    size_t _addressStart{};
    size_t _addressEnd{};
    EResRelation _resRelation{EResRelation::PRODUCER};
};  // struct ScheduledOpOneResource //

}  // namespace vpux
