#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_permute
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

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            auto outputOrder = inputs[0]->getOrder();
            auto outputShape = args.at("shape").get<mv::Shape>();

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, outputOrder));
            else
                outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, outputOrder, args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::string empty;

    }

    namespace op {

        MV_REGISTER_OP(ImplicitPermute)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_permute::inputCheckFcn)
        .setOutputDef(op_implicit_permute::outputDefFcn);

    }
}
