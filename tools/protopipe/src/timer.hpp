//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>

struct IWaitable {
    using Ptr = std::shared_ptr<IWaitable>;
    virtual void wait(std::chrono::microseconds time) = 0;
    static Ptr create();
};
