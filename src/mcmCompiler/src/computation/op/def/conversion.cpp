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

            outputs.emplace_back(":0", inputs[0]->getShape(), args.at("dType").get<mv::DType>(), inputs[0]->getOrder());

        };

    }

    namespace op {
        MV_REGISTER_OP(Conversion)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::DType>("dType")
        .setInputCheck(op_conversion::inputCheckFcn)
        .setOutputDef(op_conversion::outputDefFcn)
        .setTypeTrait({"executable"});
    }

}
