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

#include "test_model/kmb_test_base.hpp"

struct MVNTestParams final {
    MVNParams params;

    LAYER_PARAMETER(bool, across_channels);
    LAYER_PARAMETER(bool, normalize_variance);
    LAYER_PARAMETER(float, eps);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const MVNTestParams& p);
