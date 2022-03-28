//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef OPS_CONVERSION_UTILS_HPP_
#define OPS_CONVERSION_UTILS_HPP_

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"


namespace mv
{
    mv::Data::TensorIterator convertPermuteToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name,  bool /*software*/,
                                    const mv::QuantizationParams& quantParams,
                                    const mv::DType& outputTensorType,
                                    const mv::Order& outputTensorOrder);
    void correct_order_string(std::string& s,const bool& reverse=false);
    void calculate_permutation_from_orders(std::vector<unsigned>& permute_order,const std::string& old_order,const std::string& new_order);
    void calculate_permutation_from_permutes(std::vector<unsigned> &P, std::vector<unsigned> &permute_order);
}

#endif // OPS_CONVERSION_UTILS_HPP_
