#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/utils/warning_manager.hpp"

namespace mv
{

    namespace op_eltwise
    {
        const std::vector<std::string> ELTWISES = {"Add", "Subtract", "Multiply", "Divide", "Pow", "Minimum",
                                               "Maximum", "And", "SqDiff", "Equal"};

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>& args, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto eltwiseType = args.at("eltwiseType").get<std::string>();
            if(std::find(ELTWISES.begin(), ELTWISES.end(), eltwiseType) == ELTWISES.end())
            {
                errMsg = "Unsupported eltwise";
                return {false, 0};
            }

            auto inputSize = inputs.size();
            if(inputSize < 2)
            {
                errMsg = "Eltwise needs at least two inputs";
                return {false, 1};
            }

            // NOTE: Compiler assumption. It's very stupid
            // for frontend to give element wise of two populated tensors
            // so we assume that there is one unpopulated tensor and it has
            // to be in position 0
            if(inputs[0]->isPopulated())
            {
                errMsg = "Eltwise needs input 0 must be unpopulated";
                return {false, 2};
            }
                        
            /// TODO: Currently Maximum doesn't support constant input with different layout,
            /// usually a zmajor input0 vs cmajor constant input1, will cause runtime crash
            /// See ticket #EISW-13808
            if(eltwiseType == "Maximum" && inputs[1]->isPopulated()){
                const mv::Order &orderA= inputs[0]->getOrder();
                const mv::Order &orderB= inputs[1]->getOrder();
                if ( orderA != orderB){
                    errMsg= "Maximum op needs two inputs have the same order, however orderA is "
                             +                 orderA.toString()
                             + ", orderB is "+ orderB.toString() ;
                    return {false, 2};
                }
            }
            
            // SR Fix: removed input shape check,
            // handle different input-shape eltwise with broadcast eltwise now.
            return {true, 4};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto eltwiseType = args.at("eltwiseType").get<std::string>();
            //NOTE: this is done for the double tensor met, going into an eltwise for vertical fusion, by double
            //tensor we mean a tensor going into a conv and into an eltwise with different dimensions for each op
            auto input0Shape = inputs[0]->getShape();
            auto input1Shape = inputs[1]->getShape();
            std::size_t minHeight = input0Shape[mv::IO_HEIGHT_DIMENSION];
            std::size_t constant_add_unit = 1;
            if (eltwiseType == "Add")
                if (input0Shape[mv::IO_HEIGHT_DIMENSION] != input1Shape[mv::IO_HEIGHT_DIMENSION] &&
                    input0Shape[mv::IO_HEIGHT_DIMENSION] != constant_add_unit &&
                    input1Shape[mv::IO_HEIGHT_DIMENSION] != constant_add_unit)
                    minHeight = std::min(input0Shape[mv::IO_HEIGHT_DIMENSION], input1Shape[mv::IO_HEIGHT_DIMENSION]);
            auto minShape = mv::Shape({input0Shape[mv::IO_WIDTH_DIMENSION], minHeight,
                input0Shape[mv::IO_CHANNEL_DIMENSION], input0Shape[mv::IO_BATCH_DIMENSION]});
            outputs.emplace_back(":0",  minShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };
    }

    namespace op {
        MV_REGISTER_OP(Eltwise)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<std::string>("eltwiseType")
        .setInputCheck(op_eltwise::inputCheckFcn)
        .setOutputDef(op_eltwise::outputDefFcn)
        .setTypeTrait({"executable", "exposed", "optimizable"})
        .setVariableInputNum(true);

    }

}
