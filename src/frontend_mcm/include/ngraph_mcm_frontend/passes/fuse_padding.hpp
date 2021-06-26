//
// Copyright 2021 Intel Corporation.
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

#include <ngraph/pass/graph_rewrite.hpp>

//
// Merge [Pad] -> [Conv] into [Conv].
// Merge [Pad] -> [GroupConv] into [GroupConv].
// Merge [Pad] -> [MaxPool] into [MaxPool].
//

namespace ngraph {
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
}  // namespace ngraph
