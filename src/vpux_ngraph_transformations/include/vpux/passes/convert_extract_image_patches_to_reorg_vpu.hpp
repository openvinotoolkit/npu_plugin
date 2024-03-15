//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace vpux {
namespace passes {

class ConvertExtractImagePatchesToReorgYoloVPU : public ov::pass::MatcherPass {
public:
    ConvertExtractImagePatchesToReorgYoloVPU();
};

}  // namespace passes
}  // namespace vpux
