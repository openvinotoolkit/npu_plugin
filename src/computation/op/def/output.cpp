#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_output
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {
            mv::Tensor::MemoryLocation outputLocation("OUTPUT", true);
            inputs[0]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::vector<Tensor>&)
        {

        };    
    }

    namespace op {
        MV_REGISTER_OP(Output)
        .setInputs({"data"})
        .setOptionalArg<mv::DType>("precision", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_output::inputCheckFcn)
        .setOutputDef(op_output::outputDefFcn)
        .setTypeTrait({"exposed", "executable"});
    }

}
