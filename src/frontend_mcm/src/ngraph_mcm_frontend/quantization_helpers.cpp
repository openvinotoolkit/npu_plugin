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
#include <ngraph/ops.hpp>

int64_t calculateZeroPoint(float low, float high, int levels, const ngraph::element::Type& elemType) {
    IE_ASSERT((low <= 0.f) && (high >= 0.f) && (low != high));
    IE_ASSERT(levels <= 256);

    int zeroPoint = 0;

    if (elemType == ngraph::element::u8) {
        float x = -static_cast<float>(levels - 1) * low / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else if (elemType == ngraph::element::i8) {
        float x = -static_cast<float>(levels - 1) * ((high + low) * 0.5f) / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else {
        THROW_IE_EXCEPTION << "Unsupported element type " << elemType;
    }

    return zeroPoint;
}

std::vector<int64_t> calculateZeroPoints(
        const std::vector<double>& low,
        const std::vector<double>& high,
        int levels,
        const ngraph::element::Type& elemType) {
    IE_ASSERT(high.size() == low.size());

    std::vector<int64_t> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calculateZeroPoint(low[i], high[i], levels, elemType);
    }

    return out;
}

double calculateScale(float low, float high, int levels) {
    IE_ASSERT(low != high);
    IE_ASSERT(levels <= 256);

    return static_cast<double>((high - low) / static_cast<float>(levels - 1));
}

std::vector<double> calculateScales(
        const std::vector<double>& low,
        const std::vector<double>& high,
        int levels) {
    IE_ASSERT(high.size() == low.size());

    std::vector<double> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calculateScale(low[i], high[i], levels);
    }

    return out;
}

double clamp(double val, double low, double high) {
    IE_ASSERT(low <= high);
    return std::min(high, std::max(low, val));
}

bool different(double v1, double v2) {
    return std::abs(v1 - v2) > std::max(std::max(std::abs(v1), std::abs(v2)) * 0.000001, 0.000001);
}

void align_zp(float &min, float &max, const int max_levels) {
    double zp = calculateZeroPoint(min, max, max_levels, ngraph::element::u8);
    double scale = calculateScale(min, max, max_levels);
    min = static_cast<float>((0.0 - zp) * scale);
    max = static_cast<float>((max_levels - 1.0 - zp) * scale);
}

bool is_fq_agnostic(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ngraph::op::v1::Split>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::Tile>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v1::VariadicSplit>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v4::Interpolate>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v1::MaxPool>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v1::ReduceMax>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::ReorgYolo>(node) != nullptr ||
            (std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node) != nullptr &&
             std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolution>(node->output(0).get_target_inputs().begin()->get_node()->shared_from_this()) == nullptr) ||
            std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(node) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(node) != nullptr);
}

void replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const std::vector<double> &data, const std::string &name_postfix) {
    auto new_node = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, node->get_shape(), data.data());
    if (node->get_element_type() == ngraph::element::f32) {
        new_node = std::make_shared<ngraph::op::v0::Constant>(node->get_element_type(), node->get_shape(), new_node->cast_vector<float>().data());
    }
    if (node->get_element_type() == ngraph::element::f16) {
        new_node = std::make_shared<ngraph::op::v0::Constant>(node->get_element_type(), node->get_shape(), new_node->cast_vector<ngraph::float16>().data());
    }
    if (node->get_element_type() == ngraph::element::u8) {
        new_node = std::make_shared<ngraph::op::v0::Constant>(node->get_element_type(), node->get_shape(), new_node->cast_vector<unsigned char>().data());
    }

    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= different(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ngraph::replace_node(node, new_node);
}

bool replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const ngraph::element::Type_t type, const std::vector<float> &data, const std::string &name_postfix) {
    auto new_node = std::make_shared<ngraph::op::v0::Constant>(type, data.size() == 1 ? ngraph::Shape(0) : ngraph::Shape({data.size(),1,1,1}), data.data());
    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= different(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ngraph::replace_node(node, new_node);

    return changed;
}

void replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const ngraph::element::Type_t type, const float data, const std::string &name_postfix) {
    auto new_node = std::make_shared<ngraph::op::v0::Constant>(type, ngraph::Shape(0), data);
    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= different(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ngraph::replace_node(node, new_node);
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
    const auto input_values = node->input_values();
    std::vector<std::shared_ptr<ngraph::Node>> result;
    for ( auto&& iv : input_values ) {
        result.emplace_back(iv.get_node_shared_ptr());
    }
    return result;
}

std::vector<std::shared_ptr<ngraph::Node>> getAnyInputsFQ(std::shared_ptr<ngraph::Node> node) {
    std::vector<std::shared_ptr<ngraph::Node> > result;
    std::set<std::shared_ptr<ngraph::Node> > visited;
    std::stack<std::shared_ptr<ngraph::Node> > layers;

    layers.push(node);
    while (!layers.empty()) {
        auto input = layers.top();
        layers.pop();
        visited.insert(input);
        if ((dynamic_cast<ngraph::op::v0::FakeQuantize*>(input.get())) &&
            (nullptr == dynamic_cast<ngraph::op::v0::Constant*>(input->input_value(0).get_node()))) {
            result.push_back(input);
        } else {
            auto inputs = getParents(input);
            for (const auto& newInput : inputs) {
                if (!visited.count(newInput)) {
                    layers.push(newInput);
                }
            }
        }
    }
    return result;
}

std::vector<std::shared_ptr<ngraph::Node>> getInputsFQ(std::shared_ptr<ngraph::Node> node) {
    const auto parents = getParents(node);
    std::vector<std::shared_ptr<ngraph::Node>> result;
    std::vector<std::shared_ptr<ngraph::Node>> tmp;
    for ( const auto& input : parents ) {
        tmp = getAnyInputsFQ(input);
        if (tmp.size()) {
            std::move(tmp.begin(), tmp.end(), std::back_inserter(result));
            tmp.clear();
        } else
            return std::vector<std::shared_ptr<ngraph::Node>>();
    }
    return result;
}

// clang-format on
