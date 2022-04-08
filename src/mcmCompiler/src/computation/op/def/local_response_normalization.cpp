//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_local_response_normalization
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
              if (inputs[0]->getShape().ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());

                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            outputs.emplace_back(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());

        };

    }

    namespace op {
        //NOTE: Myriad X only can recieve bias and size parameters for LRN
        MV_REGISTER_OP(LocalResponseNormalization)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<unsigned>("size")
        .setArg<unsigned>("bias")
        .setInputCheck(op_local_response_normalization::inputCheckFcn)
        .setOutputDef(op_local_response_normalization::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
