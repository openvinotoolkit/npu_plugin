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
            const auto& startCoord = args.at("begin").get<mv::Shape>();
            const auto& outputSize = args.at("size").get<mv::Shape>();
            const auto& inputShape = inputs[0]->getShape();

            //TODO: support variable number of dims
            if ( (startCoord.ndims() != 4) or
                 (outputSize.ndims() != 4) or
                 (inputShape.ndims() != 4) )
            {
                errMsg = "Invalid shape at input. InputTensor - StartingCoord - OutputSize - must have a dimensionality of 4"
                        + std::to_string(inputShape.ndims()) + " - " + std::to_string(startCoord.ndims()) + " - " + std::to_string(outputSize.ndims());
                return {false, 0};
            }

            //starting coord must be inside input tensor
            // TODO: for variable number of dimensions, the "largest" tensor should be the input, and
            // we will need to check if the "sliced" dimensions exist in the inputTensor.
            for( auto dim = 0 ; dim < inputShape.ndims() ; dim++)
            {
                if( not((0 <= startCoord[dim]) and
                        (startCoord[dim] <= startCoord[dim] + outputSize[dim]) and
                        (startCoord[dim] + outputSize[dim] <= inputShape[dim])))
                {
                    errMsg = "Invalid configuration for dimension " + std::to_string(dim) + " Not able to slice : startCoord - outputSize - inputTensor"
                            + startCoord.toString() + " - " + outputSize.toString() + " " + inputShape.toString();
                    return {false, 0};
                }
            }

            return {true, 0};
        };


        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            if(args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0",args.at("size").get<mv::Shape>(),inputs[0]->getDType(),inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",args.at("size").get<mv::Shape>(),inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

            if (inputs[0]->isPopulated())
            {
                std::vector<mv::DataElement> temp(args.at("size").get<mv::Shape>().totalSize(), mv::DataElement(0));
                outputs[0].populate(temp);
            }
        };

        MV_REGISTER_OP(Slice)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("begin")
        .setArg<mv::Shape>("size")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"exposed"});
    }


}
