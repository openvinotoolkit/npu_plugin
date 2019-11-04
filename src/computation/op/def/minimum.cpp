#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_minimum
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto inputSize = inputs.size();
            if(inputSize < 2)
            {
                errMsg = "Eltwise needs at least two inputs";
                return {false, 1};
            }

            for(std::size_t i = 1; i < inputSize; ++i)
            {
                if (inputs[0]->getShape() != inputs[i]->getShape())
                {
                    errMsg = "All the inputs of eltwise ops have to share the same size";
                    return {false, 1};
                }
            }
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
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
