#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_copy
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto& inputShape = inputs[0]->getShape();

            //TODO: support variable number of dims
            if ( (inputShape.ndims() != 4) )
            {
                errMsg = "Invalid shape at input. InputTensor - must have a dimensionality of 4. has : "
                        + std::to_string(inputShape.ndims());
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
            if(args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0",inputs[0]->getShape(),dTypeToUse,inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",inputs[0]->getShape(),dTypeToUse,inputs[0]->getOrder(),args.at("quantParams").get<mv::QuantizationParams>()));
        };

    }

    namespace op {
        MV_REGISTER_OP(Copy)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_copy::inputCheckFcn)
        .setOutputDef(op_copy::outputDefFcn)
        .setTypeTrait({"exposed"});
    }


}
