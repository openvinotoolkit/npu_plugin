#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_norm
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 1)
            {
                std::stringstream err;
                err << "Incorrect number of inputs (must be 1): " << inputs.size();
                errMsg = err.str();
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


        MV_REGISTER_OP(Norm)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<double>("alpha")
        .setArg<double>("beta")
        .setArg<std::string>("region")
        .setArg<unsigned>("local_size")
        .setInputCheck(op_norm::inputCheckFcn)
        .setOutputDef(op_norm::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
