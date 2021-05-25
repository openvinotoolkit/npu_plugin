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
