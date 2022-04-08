//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_floor
    {
        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs.size() != 1) {
                errMsg = "Invalid number of inputs - must be 1, has " + std::to_string(inputs.size());
                return {false, 0};
            }
            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {
            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
        };
    }

    namespace op {
        MV_REGISTER_OP(Floor)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setInputCheck(op_floor::inputCheckFcn)
        .setOutputDef(op_floor::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }
}
