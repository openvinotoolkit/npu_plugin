#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_concat
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            // Axis be either "N", "C", "D", "H", or "W"
            auto axisToConcat = args.at("axis").get<std::string>();
            auto numericAxisToConcat = mv::Shape::getAxis(axisToConcat);

            auto inputShape0 = inputs[0]->getShape();

            if (inputShape0.ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                
                return {false, 0};
            }
            
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShapeI = inputs[i]->getShape();
            
                if (inputShapeI.ndims() != 4)
                {
                    errMsg = "Invalid shape of the input tensor (input " + std::to_string(i) + ") - must have a dimensionality of 4, "
                        " has " + std::to_string(inputShapeI.ndims());
                    return {false, 0};
                }

                // Based on concat axis, the other dimensions should match
                for(std::size_t shapeDimension = 0; shapeDimension < inputShape0.ndims(); ++shapeDimension)
                {
                    if(shapeDimension == numericAxisToConcat)
                        continue;
                    if (inputShapeI[shapeDimension] != inputShape0[shapeDimension])
                    {
                        errMsg = "Invalid shape of the input tensor (input " + std::to_string(i) + ") - inconsistent with the dimension of "
                            " the first input (input 0) ";

                        return {false, 0};
                    }
                }
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            // Axis be either "N", "C", "D", "H", or "W"
            auto axisToConcat = args.at("axis").get<std::string>();
            auto numericAxisToConcat = mv::Shape::getAxis(axisToConcat);

            std::vector<std::size_t> inputShape0(inputs[0]->getShape());
            
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShape = inputs[i]->getShape();
                inputShape0[numericAxisToConcat] += inputShape[numericAxisToConcat];
            }

            // TODO: set DType to Float16 if all inputs are from SW layers
            auto dTypeToUse = mv::DType("UInt8");

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShape0), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShape0), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

        // Default axis is channels (like for Intel Inference Engine)
        static std::string channels = "C";
    }

    namespace op {


        MV_REGISTER_OP(ImplicitConcat)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setVariableInputNum(true)
        .setOptionalArg<std::string>("axis", op_implicit_concat::channels)
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_concat::inputCheckFcn)
        .setOutputDef(op_implicit_concat::outputDefFcn);

    }
}
