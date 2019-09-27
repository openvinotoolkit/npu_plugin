#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/utils/custom_math.hpp"

namespace mv
{

    namespace op_align
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto& inputShape = inputs[0]->getShape();

            //TODO: support variable number of dims
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

            return {true, 0};
        };


        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            const auto& inputShape = inputs[0]->getShape();
            auto dim = args.at("dimension").get<std::size_t>();
            auto pad = args.at("pad").get<std::size_t>();

            auto dimPadded = mv::round_up(inputShape[dim], pad);
            auto outputShape = inputShape;
            outputShape[dim] = dimPadded;

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
        MV_REGISTER_OP(Align)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<std::size_t>("dimension", mv::IO_CHANNEL_DIMENSION)
        .setOptionalArg<std::size_t>("pad", 16)
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_align::inputCheckFcn)
        .setOutputDef(op_align::outputDefFcn)
        .setTypeTrait({"exposed"});
    }


}
