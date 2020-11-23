#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_normalize
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto eps = args.at("eps").get<double>();

            if (eps <=0 )
            {
                errMsg = "Invalid parameter: eps=" + std::to_string(eps);
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

        MV_REGISTER_OP(Normalize)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<double>("eps")
        .setOptionalArg<unsigned>("across_spatial", 0)
        .setOptionalArg<unsigned>("channel_shared", 0)
        .setInputCheck(op_normalize::inputCheckFcn)
        .setOutputDef(op_normalize::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
