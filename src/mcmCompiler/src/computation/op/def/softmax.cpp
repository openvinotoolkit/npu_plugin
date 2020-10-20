#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_softmax
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto axis = args.at("axis").get<std::string>();

            if (axis.length() != 1 || std::string("NCDHW").find(axis) == std::string::npos)
            {
                errMsg = "Invalid parameter: axis=" + axis;
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0",  inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {

        // TODO: make .setOptionalArg accept "C" instead of std::string("C")
        static std::string channels("C");

        MV_REGISTER_OP(Softmax)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<std::string>("axis", channels)
        .setInputCheck(op_softmax::inputCheckFcn)
        .setOutputDef(op_softmax::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
