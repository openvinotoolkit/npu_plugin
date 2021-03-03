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

#include "kmb_mvn_test_params.hpp"

std::ostream& operator<<(std::ostream& os, const MVNTestParams& p) {
    vpu::formatPrint(
        os, "dims: %l, across_channels: %l, normalize_variance: %l, eps: %l",
        p.dims(), p.across_channels(), p.normalize_variance(), p.eps());
    return os;
}
