#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_power
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
            if (inputs[0]->getShape().ndims() != 4)
            {
                errMsg = "Input shape ndims is not equal to 4";
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
        MV_REGISTER_OP(Power)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<double>("power", 1.0)
        .setOptionalArg<double>("scale", 1.0)
        .setOptionalArg<double>("shift", 0.0)
        .setInputCheck(op_power::inputCheckFcn)
        .setOutputDef(op_power::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }
}
