//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/constant.hpp>
#include <vector>

int64_t calculateZeroPoint(float low, float high, int levels, const ov::element::Type& elemType);

std::vector<int64_t> calculateZeroPoints(const std::vector<double>& low, const std::vector<double>& high, int levels,
                                         const ov::element::Type& elemType);

double calculateScale(float low, float high, int levels);

std::vector<double> calculateScales(const std::vector<double>& low, const std::vector<double>& high, int levels);

double clamp(double val, double low, double high);

void align_zp(float& min, float& max, const int max_levels);

bool is_fq_agnostic(const std::shared_ptr<ov::Node>& node);

void replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const std::vector<double>& data,
                             const std::string& name_postfix);
bool replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const ov::element::Type_t type,
                             const std::vector<float>& data, const std::string& name_postfix);
void replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const ov::element::Type_t type,
                             const float data, const std::string& name_postfix);

int64_t quantizeVal(double val, double scale, int64_t zeroPoint, const ov::element::Type elemType);

std::vector<int64_t> quantizeData(const ov::Shape& outShape, const ov::element::Type outElemType,
                                  const std::vector<double>& src, const ov::Shape& srcShape,
                                  const std::vector<double>& scales, const std::vector<int64_t>& zeroPoints,
                                  const ov::Shape& scalesShape);

std::vector<std::shared_ptr<ov::Node>> getInputsFQ(const std::shared_ptr<ov::Node>& node);

bool all_fqs_have_same_io_params(std::set<std::shared_ptr<ov::Node>>& fqs);
bool all_fqs_are_equal(std::vector<std::shared_ptr<ov::Node>>& fqs);
