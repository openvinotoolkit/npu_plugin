#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_fake_quantize
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto inputSize = inputs.size();
            if(inputSize < 5)
            {
                errMsg = "FakeQuantize needs at least five inputs";
                return {false, 1};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto outputShape = inputs[0]->getShape();
            auto dTypeToUse = inputs[0]->getDType();
            auto outputOrder = inputs[0]->getOrder();

            outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, outputOrder));

        };


    }

    namespace op {
        MV_REGISTER_OP(FakeQuantize)
        .setInputs({"data", "input_min", "input_max", "output_min", "output_max"})
        .setOutputs({"output"})
        .setArg<unsigned>("levels")
        .setInputCheck(op_fake_quantize::inputCheckFcn)
        .setOutputDef(op_fake_quantize::outputDefFcn)
        .setTypeTrait({"exposed", "executable"});
    }

}
