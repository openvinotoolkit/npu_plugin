#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_round
    {
        static const std::string defaultMode = "half_to_even";
        static const std::set<std::string> supportedModes = {"half_to_even", "half_away_from_zero"};

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto mode = args.at("mode").get<std::string>();
            const auto modesIter = supportedModes.find(mode);
            if (modesIter == supportedModes.end())
            {
                errMsg = "Attempt to set unsupported Round mode: " + mode
                       + ". Supported values are:";
                for (const std::string& supportedValue : supportedModes) {
                    errMsg += " ";
                    errMsg += supportedValue;
                }
                return {false, 0};
            }

            if (inputs.size() != 1) {
                errMsg = "Invalid number of inputs - must be 1, has " + std::to_string(inputs.size());
                return {false, 0};
            }

            return {true, 0}; // OK
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {
            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
        };
    }

    namespace op {
        MV_REGISTER_OP(Round)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<std::string>("mode", op_round::defaultMode)
        .setInputCheck(op_round::inputCheckFcn)
        .setOutputDef(op_round::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }
}
