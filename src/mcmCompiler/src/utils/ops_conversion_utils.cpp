//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/utils/ops_conversion_utils.hpp"

mv::Data::TensorIterator mv::convertPermuteToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name,  bool /*software*/,
                                    const mv::QuantizationParams& quantParams,
                                    const mv::DType& outputTensorType,
                                    const mv::Order& outputTensorOrder)
{
    auto order = attrs.at("order").get<mv::Order>();

    auto upaPermute = om.uPATaskPermute(name, inputs, order);
    upaPermute->setDType(outputTensorType);
    upaPermute->setQuantParams(quantParams);
    upaPermute->setOrder(outputTensorOrder);
    auto upaPermuteOp = om.getSourceOp(upaPermute);

    auto zMoutput = attrs.find("ZMoutput");
    if(zMoutput != attrs.end() && zMoutput->second == true)
        upaPermute->setOrder(mv::Order("NHWC"));

    auto vpu_in_order_str = inputs[0]->getOrder().toString();
    auto vpu_out_order_str = vpu_in_order_str;
    auto cpu_in_order_str = std::string("NCHW");
    auto cpu_out_order_str = order.toString();

    // Reverse order strings if necessary
    correct_order_string(cpu_in_order_str, true);
    correct_order_string(cpu_out_order_str, true);

    /**********************************************************************
     Example: data order="0,2,3,1"

        CPU_in                          <---          VPU_in
        order: NCHW                                   order: NHWC
        nchw_shape: (1,2,3,4)                         nhwc_shape: (1,3,4,2)


                                                            .
            |                                             .
            |   P(a,b,c,d)                                .   P(x,y,z)
            \ /  e.g., P(0,2,3,1)                         \ /  e.g., P(1,2,0)
            `                                             `


        CPU_out                        --->         VPU_out
        order: NCHW_P(a,b,c,d)                      order: NHWC_P(x,y,z)
        nchw_shape: (1,3,4,2)                       nhwc_shape: (1,4,2,3)

    **********************************************************************/

    std::vector<unsigned> po_VPU_in_to_CPU_in(3);
    std::vector<unsigned> po_CPU_in_to_CPU_out(3);
    std::vector<unsigned> po_CPU_out_to_VPU_out(3);
    std::vector<unsigned> po_VPU_in_to_VPU_out_relative = {0,1,2};
    std::vector<unsigned> po_VPU_in_to_VPU_out_xyz(3);

    // Correct order of strings if necessary (e.g., NCHW instead of WHCN)
    correct_order_string(vpu_in_order_str);
    correct_order_string(cpu_in_order_str);
    correct_order_string(cpu_out_order_str);
    correct_order_string(vpu_out_order_str);

    // Steps:
    // 1) Calculate the permute_orders for each of the 3 order transitions:
    //      - VPU_in --> CPU_in
    //      - CPU_in --> CPU_out
    //      - CPU_out (i.e., CPU_in) --> VPU_out
    calculate_permutation_from_orders(po_VPU_in_to_CPU_in, vpu_in_order_str, cpu_in_order_str);
    calculate_permutation_from_orders(po_CPU_in_to_CPU_out, cpu_in_order_str, cpu_out_order_str);
    calculate_permutation_from_orders(po_CPU_out_to_VPU_out, cpu_in_order_str, vpu_out_order_str);

    // 2) Calculate the functionally-equivalent permute_order for:
    //      - VPU_in --> VPU_out
    calculate_permutation_from_permutes(po_VPU_in_to_CPU_in, po_VPU_in_to_VPU_out_relative);
    calculate_permutation_from_permutes(po_CPU_in_to_CPU_out, po_VPU_in_to_VPU_out_relative);
    calculate_permutation_from_permutes(po_CPU_out_to_VPU_out, po_VPU_in_to_VPU_out_relative);

    upaPermuteOp->set<unsigned>("permute_order_x", po_VPU_in_to_VPU_out_relative.at(0));
    upaPermuteOp->set<unsigned>("permute_order_y", po_VPU_in_to_VPU_out_relative.at(1));
    upaPermuteOp->set<unsigned>("permute_order_z", po_VPU_in_to_VPU_out_relative.at(2));

    return upaPermute;
}

// Reverse order string if necessary
// e.g., reverse=false; in=CWHN; out=NHWC
//       reverse=true; in=NCHW; out=WHCN

void mv::correct_order_string(std::string& s,const bool& reverse)
{
    const auto N_index = (reverse) ? 3 : 0;
    if (s.at(N_index) != 'N')
        s = std::string(s.rbegin(), s.rend());
}

// Calculate P(x,y,z) from old_order & new_order
// e.g., NCHW -> NHWC  =  P(1,2,0)

void mv::calculate_permutation_from_orders(std::vector<unsigned>& permute_order,const std::string& old_order,const std::string& new_order)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (new_order.at(i + 1) == old_order.at(j + 1))
                permute_order.at(i) = j;
        }

    }
}

// Update the permute_order based on permutation P()
//
//                P()
// permute_order ----> permute_order

void mv::calculate_permutation_from_permutes(std::vector<unsigned> &P, std::vector<unsigned> &permute_order)
{
    std::vector<unsigned> permute_order_copy = {permute_order.at(0), permute_order.at(1), permute_order.at(2)};
    for (auto i = 0; i < 3; i++)
    {
        permute_order.at(i) = permute_order_copy.at(P.at(i));
    }
}