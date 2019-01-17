#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string&) -> std::pair<bool, std::size_t>
        {
            auto opType = args.at("taskOp").get<std::string>();
            std::string errMsg;
            return mv::op::OpRegistry::checkInputs(opType, inputs, args, errMsg);
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto opType = args.at("taskOp").get<std::string>();

            //WORKS, because outputs vector gets filled as in Op Ctor
            mv::op::OpRegistry::getOutputsDef(opType, inputs, args, outputs);
        };
    
        MV_REGISTER_OP(DPUTask)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable"})
        .setInputVectorTypes(true)
        .setCustomArgs(true);

    }

}
