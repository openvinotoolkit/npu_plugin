#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_resample
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto output_shape = args.at("shape").get<mv::Shape>();
            outputs.emplace_back(":0", output_shape,  inputs[0]->getDType(), inputs[0]->getOrder());
        };

        static std::string empty;

    }

    namespace op {


        MV_REGISTER_OP(ImplicitResample)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setInputCheck(op_implicit_resample::inputCheckFcn)
        .setOutputDef(op_implicit_resample::outputDefFcn);

    }
}
