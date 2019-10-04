#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_crop
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto& inputShape = inputs[0]->getShape();

            if (inputShape.ndims() != 4)
            {
                errMsg = "Invalid shape at input. InputTensor - must have a dimensionality of 4. has : " +
                        std::to_string(inputShape.ndims());
                return {false, 0};
            }

            auto dim = args.at("dimension").get<std::size_t>();
            if (dim >= inputShape.ndims())
            {
                errMsg = "Invalid dimension: " +
                        std::to_string(dim) + " is larger than input ndims " +
                        std::to_string(inputShape.ndims());
                return {false, 0};
            }

            auto cropVal = args.at("cropVal").get<std::size_t>();
            if (cropVal > inputShape[dim])
            {
                errMsg = "Invalid crop value: " +
                        std::to_string(cropVal) + " is larger than inputShape[" +
                        std::to_string(dim) + "] (" + std::to_string(inputShape[dim]) +")";
                return {false, 0};
            }

            return {true, 0};
        };


        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                std::vector<Tensor>& outputs)
        {
            const auto& inputShape = inputs[0]->getShape();
            auto dim = args.at("dimension").get<std::size_t>();
            auto cropVal = args.at("cropVal").get<std::size_t>();

            auto outputShape = inputShape;
            outputShape[dim] = cropVal;

            if(args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder(),
                        args.at("quantParams").get<mv::QuantizationParams>()));

            if (inputs[0]->hasAttr("Location"))
                outputs[0].set<Tensor::MemoryLocation>("Location",
                                inputs[0]->get<mv::Tensor::MemoryLocation>("Location"));
        };

    }

    namespace op {
        MV_REGISTER_OP(Crop)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::size_t>("cropVal")
        .setOptionalArg<std::size_t>("dimension", mv::IO_CHANNEL_DIMENSION)
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_crop::inputCheckFcn)
        .setOutputDef(op_crop::outputDefFcn)
        .setTypeTrait({"exposed"});
    }


}
