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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/quantization_helpers.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/runtime/reference/autobroadcast_binop.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <stack>
#include <numeric>
#include <algorithm>
#include <vector>

int64_t calcZeroPoint(
        double low, double high, size_t levels,
        const ngraph::element::Type& elemType) {
    IE_ASSERT(low < high);
    IE_ASSERT(levels > 1);

    int64_t zepoPoint = 0;

    if (elemType == ngraph::element::i8) {
        if ((low <= 0.0) && (high >= 0.0)) {
            const double x = -static_cast<double>(levels - 1) * ((high + low) * 0.5) / (high - low);
            zepoPoint = static_cast<int64_t>(std::ceil(x));
        } else if (low > 0.0) {
            zepoPoint = 127 - static_cast<int64_t>(levels - 1);
        } else if (high < 0.0) {
            zepoPoint = 127;
        }
    } else if (elemType == ngraph::element::u8) {
        if ((low <= 0.0) && (high >= 0.0)) {
            const double x = -static_cast<double>(levels - 1) * low / (high - low);
            zepoPoint = static_cast<int64_t>(std::ceil(x));
        } else if (low >= 0.0) {
            zepoPoint = 0;
        } else if (high <= 0.0) {
            zepoPoint = static_cast<int64_t>(levels - 1);
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported element type " << elemType;
    }

    return zepoPoint;
}

std::vector<int64_t> calcZeroPoints(
        const std::vector<double>& low,
        const std::vector<double>& high,
        size_t levels,
        const ngraph::element::Type& elemType) {
    IE_ASSERT(high.size() == low.size());

    std::vector<int64_t> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calcZeroPoint(low[i], high[i], levels, elemType);
    }

    return out;
}

double calcScale(double low, double high, size_t levels) {
    IE_ASSERT(low < high);
    IE_ASSERT(levels > 1);

    return (high - low) / (levels - 1);
}

std::vector<double> calcScales(
        const std::vector<double>& low,
        const std::vector<double>& high,
        size_t levels) {
    IE_ASSERT(high.size() == low.size());

    std::vector<double> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calcScale(low[i], high[i], levels);
    }

    return out;
}

double clamp(double val, double low, double high) {
    return std::min(high, std::max(low, val));
}

int64_t quantizeVal(
        double val, double scale, int64_t zeroPoint,
        const ngraph::element::Type elemType) {
    int64_t qVal = 0;

    if (elemType == ngraph::element::u8) {
        qVal = static_cast<int64_t>(clamp(std::round(val / scale + zeroPoint), 0, 255));
    } else {
        THROW_IE_EXCEPTION << "Unsupported element type " << elemType;
    }

    return qVal;
}

std::vector<int64_t> quantizeData(
        const ngraph::Shape& outShape,
        const ngraph::element::Type outElemType,
        const std::vector<double>& src,
        const ngraph::Shape& srcShape,
        const std::vector<double>& scales,
        const std::vector<int64_t>& zeroPoints,
        const ngraph::Shape& scalesShape) {
    const auto broadcast_spec = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY);

    std::vector<size_t> srcInds(ngraph::shape_size(srcShape));
    std::iota(srcInds.begin(), srcInds.end(), size_t(0));

    std::vector<size_t> scalesInds(ngraph::shape_size(scalesShape));
    std::iota(scalesInds.begin(), scalesInds.end(), size_t(0));

    std::vector<int64_t> out(ngraph::shape_size(outShape));

    ngraph::runtime::reference::autobroadcast_binop(
        srcInds.data(), scalesInds.data(), out.data(), srcShape, scalesShape, broadcast_spec, [&](size_t srcInd, size_t scaleInd) -> int64_t {
            const auto srcVal = src[srcInd];
            const auto scale = scales[scaleInd];
            const auto zeroPoint = zeroPoints[scaleInd];
            return quantizeVal(srcVal, scale, zeroPoint, outElemType);
        });

    return out;
}

std::vector<std::shared_ptr<ngraph::Node>> getParents(std::shared_ptr<ngraph::Node> node) {
    auto input_values = node->input_values();
    std::vector<std::shared_ptr<ngraph::Node>> result;
    for ( auto&& iv : input_values ) {
        result.emplace_back(iv.get_node_shared_ptr());
    }
    return result;
}

std::vector<std::shared_ptr<ngraph::Node>> getInputsFQ(std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::Node> > result;
    std::set<std::shared_ptr<ngraph::Node> > visited;
    std::stack<std::shared_ptr<ngraph::Node> > layers;

    auto inputs = getParents(node);
    for (auto& input : inputs) {
        layers.push(input);
    }
    while (!layers.empty()) {
        auto input = layers.top();
        layers.pop();
        visited.insert(input);
        if ((dynamic_cast<ngraph::op::v0::FakeQuantize*>(input.get())) && 
            (nullptr == dynamic_cast<ngraph::op::v0::Constant*>(input->input_value(0).get_node()))) {
            result.push_back(input);
        } else {
            auto inputs = getParents(input);
            for (auto&& newInput : inputs) {
                if (!visited.count(newInput)) {
                    layers.push(newInput);
                }
            }
        }
    }
    return result;
}

#endif
// clang-format on
