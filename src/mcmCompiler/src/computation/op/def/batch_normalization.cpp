#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_batch_normalization
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            auto inputShape = input->getShape();

            if (inputShape.ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputShape.ndims());
                return {false, 0};
            }

            auto mean = inputs[1];
            auto meanShape = mean->getShape();

            auto variance = inputs[2];
            auto varianceShape = variance->getShape();

            auto offset = inputs[3];
            auto offsetShape = offset->getShape();

            auto scale = inputs[4];
            auto scaleShape = scale->getShape();

            if (meanShape != varianceShape || meanShape != offsetShape || meanShape != offsetShape)
            {
                errMsg = "Invalid dimensionality of parameter input tensors - must have an equal dimensionality, recevied"
                    " mean " + std::to_string(meanShape.ndims()) + ", variance " + std::to_string(varianceShape.ndims()) +
                    ", offset " + std::to_string(offsetShape.ndims()) + ", scale " + std::to_string(scaleShape.ndims());
                return {false, 1};
            }

            if (inputShape != meanShape || inputShape != varianceShape || inputShape != offsetShape || inputShape != scaleShape)
            {

                if (meanShape.ndims() != 1)
                {
                    errMsg = "Invalid shape of the mean tensor (input 1) - must have a dimensionality equal to 1 or"
                        " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims());
                    return {false, 1};
                }

                if (meanShape[0] != inputShape[mv::IO_CHANNEL_DIMENSION])
                {
                    errMsg = "Invalid shape of the mean tensor (input 1) - if it has 1 dimension, it must be equal"
                        " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1]);
                    return {false, 1};
                }

                if (varianceShape.ndims() != 1)
                {
                    errMsg = "Invalid shape of the variance tensor (input 1) - must have a dimensionality equal to 1 or"
                        " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims());
                    return {false, 2};
                }

                if (varianceShape[0] != inputShape[mv::IO_CHANNEL_DIMENSION])
                {
                    errMsg = "Invalid shape of the variance tensor (input 1) - if it has 1 dimension, it must be equal"
                        " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1]);
                    return {false, 2};
                }

                if (offsetShape.ndims() != 1)
                {
                    errMsg = "Invalid shape of the offset tensor (input 1) - must have a dimensionality equal to 1 or"
                        " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims());
                    return {false, 3};
                }

                if (offsetShape[0] != inputShape[mv::IO_CHANNEL_DIMENSION])
                {
                    errMsg = "Invalid shape of the offset tensor (input 1) - if it has 1 dimension, it must be equal"
                        " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1]);
                    return {false, 3};
                }

                if (scaleShape.ndims() != 1)
                {
                    errMsg = "Invalid shape of the scale tensor (input 1) - must have a dimensionality equal to 1 or"
                        " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims());
                    return {false, 4};
                }

                if (scaleShape[0] != inputShape[mv::IO_CHANNEL_DIMENSION])
                {
                    errMsg =  "Invalid shape of the scale tensor (input 1) - if it has 1 dimension, it must be equal"
                        " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1]);
                    return {false, 4};
                }

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
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };


    }


    namespace op {

        MV_REGISTER_OP(BatchNormalization)
        .setInputs({"data", "mean", "variance", "offset", "scale"})
        .setOutputs({"output"})
        .setArg<double>("eps")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_batch_normalization::inputCheckFcn)
        .setOutputDef(op_batch_normalization::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});


    }

}
