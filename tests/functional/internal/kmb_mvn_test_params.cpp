//
// Copyright 2020 Intel Corporation.
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

#include "kmb_mvn_test_params.hpp"

std::ostream& operator<<(std::ostream& os, const MVNTestParams& p) {
    vpux::printTo(
        os, "dims: {0}, across_channels: {1}, normalize_variance: {2}, eps: {3}",
        p.dims(), p.across_channels(), p.normalize_variance(), p.eps());
    return os;
}
