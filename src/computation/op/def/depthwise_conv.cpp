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
            auto opInput = inputs[0];
            auto weights = inputs[1];
            if (opInput->getShape().ndims() != 3)
            {
                errMsg = "Shape ndims is not equal to 3";
                return {false, 0};
            }

            if (weights->getShape().ndims() != 4)
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 1};
            }

            if (opInput->getShape()[2] != weights->getShape()[3])
            {
                errMsg = "Output channel does not match the channel dimension of input " + std::to_string(opInput->getShape()[2]);
                return {false, 1};
            }

            if (weights->getShape()[2] != 1 /*weights->getShape()[2]*/)
            {
                errMsg = "Number of weights channels not match 1 " + std::to_string(opInput->getShape()[2]);
                return {false, 1};
            }
            
            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

            if (opInput->getShape()[0] + padding[0] + padding[1] < weights->getShape()[0])
            {
                errMsg = "Width exceeds padded input width " + std::to_string(opInput->getShape()[0] + padding[0] + padding[1]);
                return {false, 1};
            }

            if (opInput->getShape()[1] + padding[2] + padding[3] < weights->getShape()[1])
            {
                errMsg = "Height exceeds padded input height " + std::to_string(opInput->getShape()[1] + padding[2] + padding[3]);
                return {false, 1};
            }

            auto dilationFactor = args.at("dilationFactor").get<unsigned>();
            if (dilationFactor < 1) {

                errMsg = "Dilation factor must be greater than or equal to one";
                return {false, 1};

            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto opInput = inputs[0];
            auto weights = inputs[1];
            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();

            // Make sure that the result of subtract will not be negative
            mv::Shape outputShape({(inputs[0]->getShape()[0] + padding[0] + padding[1] - weights->getShape()[0]) / stride[0] + 1, (
                inputs[0]->getShape()[1] + padding[2] + padding[3] - weights->getShape()[1]) / stride[1] + 1, inputs[0]->getShape()[2]});

            outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    
        MV_REGISTER_OP(DepthwiseConv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("stride")
        .setOptionalArg<unsigned>("dilationFactor", 1)
        .setArg<std::array<unsigned short, 4>>("padding")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
