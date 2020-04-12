#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_output
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            if (!inputs[0]->isQuantized())
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",
                                    inputs[0]->getShape(),
                                    inputs[0]->getDType(),
                                    inputs[0]->getOrder(),
                                    inputs[0]->get<mv::QuantizationParams>("quantParams")));

        };

    }

    namespace op {
        MV_REGISTER_OP(ImplicitOutput)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_output::inputCheckFcn)
        .setOutputDef(op_implicit_output::outputDefFcn)
        .setTypeTrait({"exposed"});
    }

}
