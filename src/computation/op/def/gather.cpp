#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_gather
    {
        namespace {
            enum ShapeDesc {
                shape0D,
                shape1D,
                shape2D,
                shape3D,
                shape4D,
                shapeOverflow
            };
            ShapeDesc incrementShapeDesc(ShapeDesc sh) {
                switch(sh) {
                    case shape0D: return shape1D;
                    case shape1D: return shape2D;
                    case shape2D: return shape3D;
                    case shape3D: return shape4D;
                }
                return shapeOverflow;
            }

            ShapeDesc getShapeDesc(const Shape& sh) {
                ShapeDesc outShapeDesc = shape0D;
                for(size_t i = 0; i < sh.ndims(); ++i) {
                    if((outShapeDesc == shape0D && sh[i] > 0) || (outShapeDesc > shape0D && sh[i] > 1)) {
                        outShapeDesc = incrementShapeDesc(outShapeDesc);
                    }
                }
                return outShapeDesc;
            }
        }

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            // check inputs count
            if(inputs.size() != 2) {
                errMsg = "Invalid inputs count" + std::to_string(inputs.size());
                return {false, 1};
            }

            // check axis
            auto input = inputs[0];
            auto inputShape = input->getShape();

            auto axis  = args.at("axis").get<unsigned>();
            if (axis < 0 || axis >= inputShape.ndims()) {
                errMsg = "Invalid axis number - has to be more then 0 and less than number of dimensions - 1"
                    + std::to_string(axis);
                return {false, 1};
            }

            // check indices
            auto indices = inputs[1];
            auto indicesShape = indices->getShape();

            if(getShapeDesc(indicesShape) != shape1D) {
                errMsg = "Indices shape more then 1D is not supported yet";
                return {false, 1};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            mv::Order order(inputs[0]->getOrder());

            auto axis = 3 - args.at("axis").get<unsigned>();

            auto inputShape = inputs[0]->getShape();
            auto indicesShape = inputs[1]->getShape();

            // construct output dims
            std::vector<size_t> outputDims;
            for (int i = 0; i < axis; i++) {
                outputDims.push_back(inputShape[i]);
            }

            outputDims.push_back(indicesShape.totalSize());

            for (int i = axis + 1; i < inputShape.ndims(); i++) {
                outputDims.push_back(inputShape[i]);
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", Shape(outputDims), dTypeToUse, order));
            else
                outputs.push_back(mv::Tensor(":0", Shape(outputDims), dTypeToUse, order, args.at("quantParams").get<mv::QuantizationParams>()));
        };
    }

    namespace op {

        // Gather layer do something

        MV_REGISTER_OP(Gather)
        .setInputs({"data", "indices"})
        .setOutputs({"output"})
        .setArg<unsigned>("axis")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_gather::inputCheckFcn)
        .setOutputDef(op_gather::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
