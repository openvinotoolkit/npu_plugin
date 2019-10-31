#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_maximum
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
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputIntDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

    }



    namespace op {
        MV_REGISTER_OP(MaximumDouble)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<double>("maximum", std::numeric_limits<double>::infinity())
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_maximum::inputCheckFcn)
        .setOutputDef(op_maximum::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

        MV_REGISTER_OP(MaximumInt)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<int64_t>("maximum", 2147483647)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_maximum::inputCheckFcn)
        .setOutputDef(op_maximum::outputIntDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
