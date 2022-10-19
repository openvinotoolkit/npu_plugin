//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
