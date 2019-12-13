#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    namespace op_upa
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string&) -> std::pair<bool, std::size_t>
        {
            auto opIt = args.at("taskOp").get<std::string>();
            std::string errMsg;

            //Necessary, otherwise check on attributes fails
            auto argsBackup = args;
            argsBackup.erase("taskOp");
            return mv::op::OpRegistry::checkInputs(opIt, inputs, argsBackup, errMsg);
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto opIt = args.at("taskOp").get<std::string>();

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            mv::op::OpRegistry::getOutputsDef(opIt, inputs, args, outputs);
        };
    }
    namespace op
    {
        MV_REGISTER_OP(UPATask)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_upa::inputCheckFcn)
        .setInputCheck(false)
        .setOutputDef(op_upa::outputDefFcn)
        .setTypeTrait({"executable"})
        .setVariableInputNum(true)
        .setBaseOperation({"Dummy", "Identity", "Softmax", "Proposal", "ROIPooling", "Quantize", "Reshape",
                           "RegionYolo", "ReorgYolo", "Normalize", "Permute", "Eltwise", "Interp",
                           "DetectionOutput", "Priorbox", "Argmax", "Norm", "Resample"})
        .setExtraInputs(true);
    }

}
