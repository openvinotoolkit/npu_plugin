//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "test_model/kmb_test_base.hpp"

struct MVNTestParams final {
    MVNParams params;

    LAYER_PARAMETER(bool, across_channels);
    LAYER_PARAMETER(bool, normalize_variance);
    LAYER_PARAMETER(float, eps);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const MVNTestParams& p);
