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
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                return {false, 0};
            }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto kSize = args.at("kSize").get<std::array<unsigned short, 2>>();
            
            if (inputs[0]->getShape()[0] + padding[0] + padding[1] < kSize[0])
            {
                errMsg = "Filter kernel width (" + std::to_string(kSize[0]) + ") exceeds the padded input width ("
                    + std::to_string(inputs[0]->getShape()[0] + padding[0] + padding[1]) + ")";
                return {false, 0};
            }

            if (inputs[0]->getShape()[1] + padding[2] + padding[3] < kSize[1])
            {
                errMsg = "Filter kernel height (" + std::to_string(kSize[1]) + ") exceeds the padded input height ("
                    + std::to_string(inputs[0]->getShape()[1] + padding[2] + padding[3]) + ")";
                return {false, 0};
            }

            return {true, 0};
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto inputShape = inputs[0]->getShape();
            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();
            auto kSize = args.at("kSize").get<std::array<unsigned short, 2>>();

            Shape outputShape({(inputShape[0] + padding[0] + padding[1] - kSize[0]) / stride[0] + 1,
                (inputShape[1] + padding[2] + padding[3] - kSize[1]) / stride[1] + 1, inputShape[2]});

            outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    
        MV_REGISTER_OP(Averagepool2D)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("kSize")
        .setArg<std::array<unsigned short, 2>>("stride")
        .setArg<std::array<unsigned short, 4>>("padding")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
