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

            auto group = args.at("group").get<unsigned>();
            if (group < 1)
            {
                std::stringstream err;
                err << "Group factor must be non-zero: group=" << group;
                errMsg = err.str();
                return {false, 0};
            }

            if (inputs[0]->getShape().ndims() != 4 )
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 0};
            }

            if (inputs[1]->getShape().ndims() != 4)
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 1};
            }

//            if (inputs[0]->getShape()[IO_CHANNEL_DIMENSION] != inputs[1]->getShape()[KERNEL_INPUT_CHANNELS])
//            {
//                errMsg = "Does not match the channel dimension of input " + std::to_string(inputs[0]->getShape()[KERNEL_INPUT_CHANNELS]);
//                return {false, 1};
//            }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

            if (inputs[0]->getShape()[IO_WIDTH_DIMENSION] + padding[0] + padding[1] < inputs[1]->getShape()[KERNEL_WIDTH])
            {
                errMsg = "Width exceeds padded input width " + std::to_string(inputs[0]->getShape()[IO_WIDTH_DIMENSION] + padding[0] + padding[1]);
                return {false, 1};
            }

            if (inputs[0]->getShape()[IO_HEIGHT_DIMENSION] + padding[2] + padding[3] < inputs[1]->getShape()[KERNEL_HEIGHT])
            {
                errMsg = "Height exceeds padded input height " + std::to_string(inputs[0]->getShape()[IO_HEIGHT_DIMENSION] + padding[2] + padding[3]);
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
            auto group = args.at("group").get<unsigned>();

            // TODO: Please take dilation factor into account!
            // Make sure that the result of subtract will not be negative
            auto W = (inputs[0]->getShape()[IO_WIDTH_DIMENSION] + padding[0] + padding[1] - inputs[1]->getShape()[KERNEL_WIDTH]) / stride[0] + 1;
            auto H = (inputs[0]->getShape()[IO_HEIGHT_DIMENSION] + padding[2] + padding[3] - inputs[1]->getShape()[1]) / stride[1] + 1;
            auto C =  inputs[1]->getShape()[KERNEL_OUTPUT_CHANNELS] * group;
            auto N = inputs[0]->getShape()[IO_BATCH_DIMENSION];

            mv::Shape outputShape({W, H, C, N});

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

        MV_REGISTER_OP(Conv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("stride")
        .setArg<std::array<unsigned short, 4>>("padding")
        .setOptionalArg<unsigned>("dilationFactor", 1)
        .setOptionalArg<unsigned>("group", 1)
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
