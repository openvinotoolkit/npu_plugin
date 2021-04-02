#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_ctc_greedy_decoder_seq_len
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs.size() != 2 && inputs.size() != 3)
            {
                errMsg = "CTCGreedyDecoderSeqLen should have 2 or 3 inputs.";
                return {false, 0};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            const auto input0 = inputs[0]->getShape();

            const auto output0 = mv::Shape{input0[IO_HEIGHT_DIMENSION], input0[IO_CHANNEL_DIMENSION], 1, 1};
            const auto output1 = mv::Shape{input0[IO_CHANNEL_DIMENSION], 1, 1, 1};

            outputs.emplace_back(":0", output0, inputs[1]->getDType(), inputs[0]->getOrder());
            outputs.emplace_back(":1", output1, inputs[1]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(CTCGreedyDecoderSeqLen)
        .setInputs({"data", "seqLen", "blankIndex"})
        .setOutputs({"output"})
        .setArg<bool>("merge_repeated")
        .setInputCheck(op_ctc_greedy_decoder_seq_len::inputCheckFcn)
        .setOutputDef(op_ctc_greedy_decoder_seq_len::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
