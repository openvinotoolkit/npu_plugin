//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

//
// Merge [Pad] -> [Conv] into [Conv].
// Merge [Pad] -> [GroupConv] into [GroupConv].
// Merge [Pad] -> [MaxPool] into [MaxPool].
//

namespace vpux {
namespace pass {

class FusePadding : public ngraph::pass::MatcherPass {
public:
    FusePadding();

protected:
    template <class T>
    bool setPadding(const size_t rank, const T& pads_begin, const T& pads_end,
                    const std::function<void(const T&, const T&)>& setter);
};

}  // namespace pass
}  // namespace vpux
