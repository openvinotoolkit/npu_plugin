//
// Copyright (C) 2022 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIPRegMapped/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

namespace {

const uint32_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

static const std::unordered_map<DimsOrder, std::vector<float>> orderMapping = {
        {DimsOrder::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
        {DimsOrder::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::C, {DIM_C}},
        {DimsOrder::CHW, {DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NC, {DIM_N, DIM_C}},
};

}  // namespace

vpux::VPUIPRegMapped::NetworkDescription::NetworkDescription(std::vector<char> blob)
        : _compiledNetwork(std::move(blob)) {
    VPUX_THROW_UNLESS(!_compiledNetwork.empty(), "Got NULL pointer");
}
