#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_prelu
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs[1]->getShape().ndims() != 1)
            {
                errMsg = "Incorrect shape " + inputs[1]->getShape().toString() + " of slope (must be a vector)";
                return {false, 0};
            }

            if (inputs[0]->getShape()[IO_CHANNEL_DIMENSION] != inputs[1]->getShape()[0])
            {
                errMsg = "Mismatch in channels dimensions between input (" + std::to_string(inputs[0]->getShape()[-1])
                    + ") and slope (" + std::to_string(inputs[0]->getShape()[0]) + ")";
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
        MV_REGISTER_OP(Prelu)
        .setInputs({"data", "slope"})
        .setOutputs({"output"})
        .setInputCheck(op_prelu::inputCheckFcn)
        .setOutputDef(op_prelu::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
