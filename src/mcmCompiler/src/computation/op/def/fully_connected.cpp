#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_fully_connected
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            //
            // FullyConnected layer has the following specification:
            //
            //     FC: [N, IC] x [OC, IC] = [N, OC]
            //
            // In IRv10+ it is represented as the following generic MatMul operation:
            //
            //     MatMul: [N, IC] {​​​​​​​​​​​​​transpose_a=false}​​​​​​​​​​​​​ x [OC, IC] {​​​​​​​​​​​​​transpose_b=true}​​​​​​​​​​​​​ = [N, OC]
            //
            // The conversion from MatMul to FullyConnected is done by nGraph common transformations.
            // So, the FullyConnected has no transpose attributes, they are part of MatMul operation only.
            //
            // MCM frontend converts 2D tensors to 4D (by inserting extra 1 to shape) due to issues in MCM compiler.
            // So after the MCM fronent the FullyConnected will be represented as:
            //
            //     FC: [N, IC, 1, 1] x [OC, IC, 1, 1] = [N, OC, 1, 1]
            //
            // Either with 2D weigths tensor:
            //
            //     FC: [N, IC, 1, 1] x [OC, IC] = [N, OC, 1, 1]
            //
            // Please note also, that MCM compiler reverts tensor dims, i.e. [N, C, H, W] becomes [W, H, C, N].
            // So MCM calls FC with tensors like:
            //
            //     FC (MCM): [1, 1, IC, N] x [1, 1, IC, OC] = [1, 1, OC, N]
            //
            // Either with 2D weigths tensor:
            //
            //     FC (MCM): [1, 1, IC, N] x [IC, OC] = [1, 1, OC, N]
            //

            const int inputsNum = inputs.size();

            if (inputsNum != 2)
            {
                errMsg = "Invalid number of inputs: " + std::to_string(inputsNum)
                       + ", but there must be 2 inputs";
                return {false, 0};
            }

            const mv::Shape& shapeA = inputs[0]->getShape();
            const mv::Shape& shapeB = inputs[1]->getShape();

            const int nDimsA = shapeA.ndims(); // 4D always
            const int nDimsB = shapeB.ndims(); // 4D, or 2D

            if (nDimsA != 4)
            {
                errMsg = "Inputs tensor must be 4-dimensional: but input shape = "
                       +                        shapeA.toString()
                       + ", weights shape = " + shapeB.toString();
                return {false, 0};
            }

            if (nDimsB != 2 && nDimsB != 4)
            {
                errMsg = "Weights tensor must be 2- or 4-dimensional: but input shape = "
                       +                        shapeA.toString()
                       + ", weights shape = " + shapeB.toString();
                return {false, 0};
            }

            if (nDimsA == 4)
            {
                if (shapeA[0] != 1 || shapeA[1] != 1)
                {
                    errMsg = "Inputs tensor, if it is 4D, must have shape like {1, 1, IC, N}: but input shape = "
                        +                        shapeA.toString()
                        + ", weights shape = " + shapeB.toString();
                    return {false, 0};
                }
            }

            if (nDimsB == 4)
            {
                if (shapeB[0] != 1 || shapeB[1] != 1)
                {
                    errMsg = "Weights tensor, if it is 4D, must have shape like {1, 1, IC, OC}: but input shape = "
                        +                        shapeA.toString()
                        + ", weights shape = " + shapeB.toString();
                    return {false, 0};
                }
            }

            // Matching dimensions of tensors A and B
            const int matchDimA = shapeA[nDimsA - 2];
            const int matchDimB = shapeB[nDimsB - 2];

            if (matchDimA != matchDimB)
            {
                errMsg = "Inconsistent tensor shapes: input shape = "
                       +                        shapeA.toString()
                       + ", weights shape = " + shapeB.toString()
                       + ", input match dim = "   + std::to_string(matchDimA)
                       + ", weights match dim = " + std::to_string(matchDimB);
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            //
            // Fully Conected layer is called like:
            //
            //     FC (MCM): [1, 1, IC, N] x [1, 1, IC, OC] = [1, 1, OC, N]
            //
            // Either with 2D weights tensor, like:
            //
            //     FC (MCM): [1, 1, IC, N] x [IC, OC] = [1, 1, OC, N]
            //

            const mv::Shape& shapeA = inputs[0]->getShape();
            const mv::Shape& shapeB = inputs[1]->getShape();

            const int nDimsA = shapeA.ndims(); // 4D always
            const int nDimsB = shapeB.ndims(); // 4D, or 2D

            const size_t N  = shapeA[nDimsA - 1];
            const size_t OC = shapeB[nDimsB - 1];

            const mv::Shape shape = {1, 1, OC, N};
            outputs.emplace_back(":0", shape, inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(FullyConnected)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setInputCheck(op_fully_connected::inputCheckFcn)
        .setOutputDef(op_fully_connected::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
