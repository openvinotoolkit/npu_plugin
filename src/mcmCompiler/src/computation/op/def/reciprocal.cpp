#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_reciprocal
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
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

            auto outputShape = inputs[0]->getShape();

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

    }

    namespace op {

        MV_REGISTER_OP(Reciprocal)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_reciprocal::inputCheckFcn)
        .setOutputDef(op_reciprocal::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
