#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_divide
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            if (inputs[0]->getShape() != inputs[1]->getShape())
            {
                errMsg = "Does not match the data0 shape " + inputs[1]->getShape().toString();
                return {false, 1};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    }

    namespace op {
        MV_REGISTER_OP(Divide)
        .setInputs({"data0", "data1"})
        .setOutputs({"output"})
        .setInputCheck(op_divide::inputCheckFcn)
        .setOutputDef(op_divide::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }
}
