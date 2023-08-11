//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/utils/core/checked_cast.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <string>
#include <vector>

#define PRETTY_PARAM(name, type)                                                 \
    class name {                                                                 \
    public:                                                                      \
        using paramType = type;                                                  \
        name(paramType arg = paramType()): val_(arg) {                           \
        }                                                                        \
        operator paramType() const {                                             \
            return val_;                                                         \
        }                                                                        \
                                                                                 \
    private:                                                                     \
        paramType val_;                                                          \
    };                                                                           \
    static inline void PrintTo(name param, ::std::ostream* os) {                 \
        *os << #name ": " << ::testing::PrintToString((name::paramType)(param)); \
    }

PRETTY_PARAM(Device, std::string);

class StaticShape {
public:
    StaticShape(ov::Shape arg): staticShape_(arg) {
    }
    operator ov::Shape() const {
        return staticShape_;
    }

    operator ov::test::InputShape() const {
        return ov::test::InputShape({}, std::vector<ov::Shape>{staticShape_});
    }

private:
    ov::Shape staticShape_;
};

static inline void PrintTo(StaticShape param, ::std::ostream* os) {
    *os << ::testing::PrintToString((ov::Shape)(param));
}

template <typename... Dims>
ov::Shape makeShape(Dims... dims) {
    return ov::Shape{vpux::checked_cast<size_t>(static_cast<int>(dims))...};
}
