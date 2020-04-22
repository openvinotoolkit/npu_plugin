#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/tiling.hpp"

namespace mv
{

    namespace op_conv
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto kernel = inputs[1];
            auto kernelShape = (kernel->hasAttr("OriginalShape")) ?
                                    (kernel->get<mv::Shape>("OriginalShape")) :
                                    (kernel->getShape());

            auto data = inputs[0];
            auto dataShape = inputs[0]->getShape();

            auto group = args.at("group").get<unsigned>();
            if (group < 1)
            {
                std::stringstream err;
                err << "Group factor must be non-zero: group=" << group;
                errMsg = err.str();
                return {false, 0};
            }

            if (dataShape.ndims() != 4 )
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 0};
            }

            if (kernelShape.ndims() != 4)
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 1};
            }

           if (group == 1)
           {
               if (dataShape[IO_CHANNEL_DIMENSION] != kernelShape[KERNEL_INPUT_CHANNELS])
               {
                   errMsg = "Does not match the channel dimension of input " + std::to_string(dataShape[KERNEL_INPUT_CHANNELS]);
                   return {false, 1};
               }
           }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

            if (dataShape[IO_WIDTH_DIMENSION] + padding[0] + padding[1] < kernelShape[KERNEL_WIDTH])
            {
                errMsg = "Width exceeds padded input width " + std::to_string(dataShape[IO_WIDTH_DIMENSION] + padding[0] + padding[1]);
                return {false, 1};
            }

            if (dataShape[IO_HEIGHT_DIMENSION] + padding[2] + padding[3] < kernelShape[KERNEL_HEIGHT])
            {
                errMsg = "Height exceeds padded input height " + std::to_string(dataShape[IO_HEIGHT_DIMENSION] + padding[2] + padding[3]);
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

            auto kernel = inputs[1];
            auto kernelShape = (kernel->hasAttr("OriginalShape")) ?
                                    (kernel->get<mv::Shape>("OriginalShape")) :
                                    (kernel->getShape());

            auto data = inputs[0];
            auto dataShape = inputs[0]->getShape();

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();
            auto group = args.at("group").get<unsigned>();

            // TODO: dilation factor must be per kernel dimension
            auto dilationFactor = args.at("dilationFactor").get<unsigned>();
            auto dilated_kernel_w = (kernelShape[KERNEL_WIDTH] - 1) * dilationFactor + 1;
            auto dilated_kernel_h = (kernelShape[KERNEL_HEIGHT] - 1) * dilationFactor + 1;

            auto W = Tiling::inferOutputSize(dataShape[IO_WIDTH_DIMENSION], padding[0], padding[1], dilated_kernel_w, stride[0]);
            auto H = Tiling::inferOutputSize(dataShape[IO_HEIGHT_DIMENSION], padding[2], padding[3], dilated_kernel_h, stride[1]);
            auto C = kernelShape[KERNEL_OUTPUT_CHANNELS] * group;
            auto N = dataShape[IO_BATCH_DIMENSION];

            mv::Shape outputShape({W, H, C, N});

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, data->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };


    }

    namespace op {
        MV_REGISTER_OP(Conv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("stride")
        .setArg<std::array<unsigned short, 4>>("padding")
        .setOptionalArg<unsigned>("dilationFactor", 1)
        .setOptionalArg<unsigned>("group", 1)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_conv::inputCheckFcn)
        .setOutputDef(op_conv::outputDefFcn)
        .setTypeTrait({"executable", "exposed", "optimizable"});


    }

}
