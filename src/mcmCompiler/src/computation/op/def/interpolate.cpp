#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_interpolate
    {
        static const std::set<std::string> supportedInterpolations = {"linear", "linear_onnx", "nearest"};
        static const std::set<std::string> supportedCoordinateTransformMode = {"half_pixel", "asymmetric", "align_corners",
                                                                               "pytorch_half_pixel", "tf_half_pixel_for_nn"};
        static const std::set<std::string> supportedNearestMode = {"simple", "round_prefer_floor", "floor", "ceil", "round_prefer_ceil"};
        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto interpolation = args.at("mode").get<std::string>();
            auto coord_transform_mode = args.at("coordinate_transformation_mode").get<std::string>();
            auto nearest_mode = args.at("nearest_mode").get<std::string>();

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
            std::set<std::string>::const_iterator coordIter = supportedCoordinateTransformMode.find(coord_transform_mode);
            if (coordIter == supportedCoordinateTransformMode.end())
            {
                errMsg = "Attempt to set unsupported coordinate transformation mode: " +
                         interpolation + ". Supported values are: ";
                for (const std::string& supportedValue : supportedCoordinateTransformMode) {
                    errMsg += supportedValue + " ";
                }

                return {false, 0};
            }
            std::set<std::string>::const_iterator nearIter = supportedNearestMode.find(nearest_mode);
            if (nearIter == supportedNearestMode.end())
            {
                errMsg = "Attempt to set unsupported nearest mode: " +
                         interpolation + ". Supported values are: ";
                for (const std::string& supportedValue : supportedNearestMode) {
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
            auto outputShape = args.at("output_shape").get<mv::Shape>();

            outputs.emplace_back(":0",  outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(Interpolate)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("output_shape")
        .setOptionalArg<std::string>("mode", std::string("nearest"))
        .setOptionalArg<std::string>("nearest_mode", std::string("round_prefer_floor"))
        .setOptionalArg<std::string>("coordinate_transformation_mode", std::string("half_pixel"))
        .setOptionalArg<bool>("align_corners", false)
        .setOptionalArg<bool>("antialias", false)
        .setInputCheck(op_interpolate::inputCheckFcn)
        .setOutputDef(op_interpolate::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
