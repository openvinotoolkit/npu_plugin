#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_softmax
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto axis = args.at("axis").get<std::string>();

            if (axis.length() != 1 || std::string("NCDHW").find(axis) == std::string::npos)
            {
                errMsg = "Invalid parameter: axis=" + axis;
                return {false, 0};
            }

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
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

    }

    namespace op {

        // TODO: make .setOptionalArg accept "C" instead of std::string("C")
        static std::string channels("C");

        MV_REGISTER_OP(Softmax)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<std::string>("axis", channels)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_softmax::inputCheckFcn)
        .setOutputDef(op_softmax::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
