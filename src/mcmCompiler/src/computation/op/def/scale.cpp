#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_scale
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            auto inputShape = input->getShape();
            auto scales = inputs[1];
            auto scalesShape = scales->getShape();

            if (scalesShape.ndims() != 1)
            {
                errMsg = "Invalid shape of scales tensor (input 1) - has to be 1-dimensional, received "
                    + std::to_string(scalesShape.ndims());
                return {false, 1};
            }

            if (inputShape[IO_CHANNEL_DIMENSION] != scalesShape[0])
            {
                errMsg = "Invalid shape of scales tensor (input 1) - the dimension has to equal to the last dimension"
                    " of the input tensor which is " + std::to_string(inputShape[-1]);
                return {false, 1};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0",  inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {

        MV_REGISTER_OP(Scale)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setInputCheck(op_scale::inputCheckFcn)
        .setOutputDef(op_scale::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
