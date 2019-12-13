#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_resample
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            // Notes on 2 types of Resample:
            // - Resample (Type 1) Layer - factor specifies a scale factor for output height and width.
            // - Resample (Type 2) Layer - factor parameter is ignored; 1D blob describing output shape is used instead.
            //
            // Therefore, App must pre-calculate & specify the output shape.
            //
            auto outputShape = args.at("output_shape").get<mv::Shape>();

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::string empty;

    }

    namespace op {

        MV_REGISTER_OP(Resample)
        .setInputs({"input"})
        .setOutputs({"output"})
        .setArg<std::string>("interpolation")
        .setArg<int64_t>("antialias")
        .setArg<mv::Shape>("output_shape")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_resample::inputCheckFcn)
        .setOutputDef(op_resample::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
