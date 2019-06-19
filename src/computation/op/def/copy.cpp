#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
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
            const auto& inputShape = inputs[0]->getShape();
            if(args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0",inputShape,inputs[0]->getDType(),inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",inputShape,inputs[0]->getDType(),inputs[0]->getOrder(),args.at("quantParams").get<mv::QuantizationParams>()));
        };

        MV_REGISTER_OP(Copy)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable","exposed"});
    }


}
