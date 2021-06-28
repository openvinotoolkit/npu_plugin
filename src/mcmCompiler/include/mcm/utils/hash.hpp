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
#include "include/mcm/computation/op/op.hpp"

typedef const mv::Op* operation_t;


// The purpose of this custom hash is primarly to enusure that data-structures in the scheduler
// are processed in the same order across platforms to enable consistent complation of blobs 
struct djb2Hash
{
    std::size_t operator()(std::string const& input) const noexcept
    {
        size_t hash;
        size_t c;

        hash = 5381;
        for (auto x : input) {
            c = x;
            hash = ((hash << 5) + hash) + c;
        }
        return hash;
    }
};

struct operation_comparator_t {
    bool operator()(const operation_t& op1, const operation_t& op2) const {

        auto hash1 = djb2Hash()(op1->getName());
        auto hash2 = djb2Hash()(op2->getName());
     
        return hash1 < hash2;
    }
    
    bool operator()(const std::string& op1, const std::string& op2) const {

        auto hash1 = djb2Hash()(op1);
        auto hash2 = djb2Hash()(op2);
     
        return hash1 < hash2;
    }
};
