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
            // Axis be either "N", "C", "D", "H", or "W"
            auto axis = args.at("axis").get<std::string>();
            auto axid = std::string("NCDHW").find(axis);

            auto order0 = inputs[0]->getOrder().toString();
            auto order1 = inputs[1]->getOrder().toString();
            auto found0 = order0.find(axis);
            auto found1 = order1.find(axis);

            if (found0 == std::string::npos ||
                found1 == std::string::npos ||
                found0 != found1            ||
                axis.length() != 1          ||
                axid == std::string::npos)
            {
                std::stringstream err;
                err << "Axis incorrect or unknown: axis=" << axis
                                           << ", order0=" << order0
                                           << ", order1=" << order1;
                errMsg = err.str();
                return {false, 0};
            }

            // TODO: un-hardcode limitation about "CHW" order
            // Note: code below assumes inputs order is "CHW"

            if (inputs[0]->getShape().ndims() != 3)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                
                return {false, 0};
            }
            
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShape = inputs[i]->getShape();
            
                if (inputShape.ndims() != 3)
                {
                    errMsg = "Invalid shape of the input tensor (input " + std::to_string(i) + ") - must have a dimensionality of 3, "
                        " has " + std::to_string(inputShape.ndims());
                    return {false, 0};
                }
                
                // TODO: based on concat axis, the other dimensions should match  
                if (inputShape[0] != inputs[0]->getShape()[0] || inputShape[1] != inputs[0]->getShape()[1]) 
                {
                    errMsg = "Invalid shape of the input tensor (input " + std::to_string(i) + ") - inconsistent with the dimension of "
                        " the first input (input 0) ";

                    return {false, 0};
                }
        
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {
            std::size_t lastDim = inputs[0]->getShape()[2];
            
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShape = inputs[i]->getShape();
                lastDim += inputShape[2];
            }
            
            outputs.push_back(mv::Tensor(":0", {inputs[0]->getShape()[0], inputs[0]->getShape()[1], lastDim}, inputs[0]->getDType(), inputs[0]->getOrder()));
        };

        // TODO: Introduce enum mv::Axis {N, C, D, H, W} type for axis
        // Default axis is channels (like for Intel Inference Engine)
        static std::string channels = "C";

        MV_REGISTER_OP(Concat)
        .setInputs({"data0", "data1"})
        .setOutputs({"output"})
        .setOptionalArg<std::string>("axis", channels)
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
