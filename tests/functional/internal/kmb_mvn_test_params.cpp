//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "kmb_mvn_test_params.hpp"

std::ostream& operator<<(std::ostream& os, const MVNTestParams& p) {
    vpux::printTo(os, "dims: {0}, across_channels: {1}, normalize_variance: {2}, eps: {3}", p.dims(),
                  p.across_channels(), p.normalize_variance(), p.eps());
    return os;
}
