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

#pragma once

// clang-format off

#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/op/constant.hpp>
#include <vector>

int64_t calculateZeroPoint(float low, float high, int levels, const ngraph::element::Type& elemType);

std::vector<int64_t> calculateZeroPoints(
        const std::vector<double>& low,
        const std::vector<double>& high,
        int levels,
        const ngraph::element::Type& elemType);

double calculateScale(float low, float high, int levels);

std::vector<double> calculateScales(
        const std::vector<double>& low,
        const std::vector<double>& high,
        int levels);

double clamp(double val, double low, double high);

bool different(double v1, double v2);

void align_zp(float &min, float &max, const int max_levels);

bool is_fq_agnostic(const std::shared_ptr<ngraph::Node>& node);

void replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const std::vector<double> &data, const std::string &name_postfix);
bool replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const ngraph::element::Type_t type, const std::vector<float> &data, const std::string &name_postfix);
void replace_node_if_changed(const std::shared_ptr<ngraph::op::v0::Constant>& node, const ngraph::element::Type_t type, const float data, const std::string &name_postfix);

int64_t quantizeVal(
        double val, double scale, int64_t zeroPoint,
        const ngraph::element::Type elemType);

std::vector<int64_t> quantizeData(
        const ngraph::Shape& outShape,
        const ngraph::element::Type outElemType,
        const std::vector<double>& src,
        const ngraph::Shape& srcShape,
        const std::vector<double>& scales,
        const std::vector<int64_t>& zeroPoints,
        const ngraph::Shape& scalesShape);

std::vector<std::shared_ptr<ngraph::Node>> getInputsFQ(std::shared_ptr<ngraph::Node> node);

// clang-format on
