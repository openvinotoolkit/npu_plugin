#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_ctcdecoder
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            auto inputShape = input->getShape();
            auto seq = inputs[1];
            auto seqShape = seq->getShape();

            if (seqShape.ndims() != 2)
            {
                errMsg = "Invalid shape of seq tensor (input 1) - has to be 2-dimensional, received "
                    + std::to_string(seqShape.ndims());
                return {false, 1};
            }

            if (inputShape[mv::IO_CHANNEL_DIMENSION] != seqShape[0])
            {
                errMsg = "Invalid shape of seq tensor (input 1) - the dimension has to equal to the last dimension"
                    " of the input tensor which is " + std::to_string(inputShape[-1]);
                return {false, 2};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&args, std::vector<Tensor>& outputs)
        {
            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();
            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
        };

    }

    namespace op {
        MV_REGISTER_OP(CTCDecoder)
        .setInputs({"data", "seq"})
        .setOutputs({"output"})
        .setArg<bool>("ctc_merge_repeated")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_ctcdecoder::inputCheckFcn)
        .setOutputDef(op_ctcdecoder::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
