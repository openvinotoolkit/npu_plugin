#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_conversion
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), args.at("order").get<Order>()));

        };

    }

    namespace op {
        MV_REGISTER_OP(Conversion)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Order>("order")
        .setInputCheck(op_conversion::inputCheckFcn)
        .setOutputDef(op_conversion::outputDefFcn)
        .setTypeTrait({"executable"});
    }

}
