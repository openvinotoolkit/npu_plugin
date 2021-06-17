#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/utils/warning_manager.hpp"

namespace mv
{

    namespace op_reverse_sequence
    {
        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>& args, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto inputSize = inputs.size();
            if(inputSize < 2)
            {
                errMsg = "Reverse sequence needs two inputs";
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0",  inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());
        };
    }

    namespace op {
        MV_REGISTER_OP(ReverseSequence)
        .setInputs({"data", "seq_length"})
        .setOutputs({"output"})
        .setArg<int64_t>("seq_axis")
        .setArg<int64_t>("batch_axis")
        .setInputCheck(op_reverse_sequence::inputCheckFcn)
        .setOutputDef(op_reverse_sequence::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
