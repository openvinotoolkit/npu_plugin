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

            if (inputs[0]->getShape().ndims() != 3)
            {
                errMsg = "Shape ndims is not equal to 3";
                return {false, 0};
            }

            if (inputs[1]->getShape().ndims() != 4)
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 1};
            }

            if (inputs[0]->getShape()[2] != inputs[1]->getShape()[2])
            {
                errMsg = "Does not match the channel dimension of input " + std::to_string(inputs[0]->getShape()[2]);
                return {false, 1};
            }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

            if (inputs[0]->getShape()[0] + padding[0] + padding[1] < inputs[1]->getShape()[0])
            {
                errMsg = "Width exceeds padded input width " + std::to_string(inputs[0]->getShape()[0] + padding[0] + padding[1]);
                return {false, 1};
            }

            if (inputs[0]->getShape()[1] + padding[2] + padding[3] < inputs[1]->getShape()[1])
            {
                errMsg = "Height exceeds padded input height " + std::to_string(inputs[0]->getShape()[1] + padding[2] + padding[3]);
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

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();
           
            // Make sure that the result of subtract will not be negative
            mv::Shape outputShape({(inputs[0]->getShape()[0] + padding[0] + padding[1] - inputs[1]->getShape()[0]) / stride[0] + 1, (
                inputs[0]->getShape()[1] + padding[2] + padding[3] - inputs[1]->getShape()[1]) / stride[1] + 1, inputs[1]->getShape()[3]});

            outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));
        
        };
    
        MV_REGISTER_OP(Conv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("stride")
        .setOptionalArg<unsigned>("dilationFactor", 1)
        .setArg<std::array<unsigned short, 4>>("padding")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed", "automatic_api"});

    }

}
