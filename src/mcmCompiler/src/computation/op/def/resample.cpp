
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_resample
    {
        static const std::set<std::string> supportedInterpolations = {"BILINEAR", "BICUBIC", "NEAREST"};

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto interpolation = args.at("interpolation").get<std::string>();
            std::set<std::string>::const_iterator interpolationIter = supportedInterpolations.find(interpolation);
            if (interpolationIter == supportedInterpolations.end())
            {
                errMsg = "Attempt to set unsupported interpolation: " +
                         interpolation + ". Supported values are: ";
                for (const std::string& supportedValue : supportedInterpolations) {
                    errMsg += supportedValue + " ";
                }

                return {false, 0};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            // Notes on 2 types of Resample:
            // - Resample (Type 1) Layer - factor specifies a scale factor for output height and width.
            // - Resample (Type 2) Layer - factor parameter is ignored; 1D blob describing output shape is used instead.
            //
            // Therefore, App must pre-calculate & specify the output shape.
            //
            auto outputShape = args.at("output_shape").get<mv::Shape>();

            outputs.emplace_back(":0",  outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };

        static std::string empty;

    }

    namespace op {

        MV_REGISTER_OP(Resample)
        .setInputs({"input"})
        .setOutputs({"output"})
        .setArg<std::string>("interpolation")
        .setArg<bool>("antialias")
        .setArg<mv::Shape>("output_shape")
        .setInputCheck(op_resample::inputCheckFcn)
        .setOutputDef(op_resample::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
