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
    vpu::formatPrint(
        os, "dims: %l, across_channels: %l, normalize_variance: %l, eps: %l",
        p.dims(), p.across_channels(), p.normalize_variance(), p.eps());
    return os;
}
