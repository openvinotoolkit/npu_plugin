#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_priorbox
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 6)
            {
                std::stringstream err;
                err << "Incorrect number of inputs (must be 6): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto input = inputs[0];
            auto outputOrder = input->getOrder();
            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  inputShape, dTypeToUse, outputOrder));
            else
                outputs.push_back(mv::Tensor(":0",  inputShape, dTypeToUse, outputOrder, args.at("quantParams").get<mv::QuantizationParams>()));

        };
    }

    namespace op
    {
        MV_REGISTER_OP(Priorbox)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<unsigned>("flip")
        .setArg<unsigned>("clip")
        .setArg<double>("step_w")
        .setArg<double>("step_h")
        .setArg<double>("offset")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_priorbox::inputCheckFcn)
        .setOutputDef(op_priorbox::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);
    }

}
