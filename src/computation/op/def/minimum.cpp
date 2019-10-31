#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_minimum
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {
            // TODO: modify for multiple inputs
            if (inputs[0]->getShape() != inputs[1]->getShape())
            {
                errMsg = "Does not match the data0 shape " + inputs[1]->getShape().toString();
                return {false, 1};
            }
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto input0Quantized = inputs[0]->hasAttr("quantParams");
            if (!input0Quantized)
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder(), inputs[0]->get<mv::QuantizationParams>("quantParams")));

        };

    }



    namespace op {

        MV_REGISTER_OP(Minimum)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setInputCheck(op_minimum::inputCheckFcn)
        .setOutputDef(op_minimum::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);
    }

}
