//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "test_model/kmb_test_base.hpp"

using namespace InferenceEngine;

struct ProfilingTaskCheckParams final {
    std::set<std::string> allowedTaskExecTypes;  // SW, DMA, DPU
    std::map<std::string, std::tuple<unsigned, unsigned, unsigned>>
            layerExecTimes;  // expected minimal execution cpu time and expected maximal execution cpu and real times
                             // [us]
};

//
// ProfilingTestBase
//

class ProfilingTestBase : public KmbLayerTestBase {
    using NetworkBuilder = std::function<void(TestNetwork& testNet)>;

public:
    void runTest(RunProfilingParams profilingRunParams, ProfilingTaskCheckParams profilingTestParams);
};
