#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_mvn
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto eps = args.at("eps").get<double>();
            if (eps <= 0)
            {
                std::stringstream err;
                err << "Invalid eps value (must be a positive floating-point number): " << eps;
                errMsg = err.str();
                return {false, 0};
            }
            if (inputs[0]->getShape().ndims() > 4)
            {
                std::stringstream err;
                err << "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                errMsg = err.str();
                return {false, 0};
            }
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {

        MV_REGISTER_OP(MVN)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<bool>("across_channels")
        .setArg<bool>("normalize_variance")
        .setArg<double>("eps")
        .setInputCheck(op_mvn::inputCheckFcn)
        .setOutputDef(op_mvn::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
