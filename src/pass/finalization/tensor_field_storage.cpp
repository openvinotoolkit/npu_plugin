#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void tensorFieldStorageFn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TensorFieldStorage)
        .setFunc(tensorFieldStorageFn)
        .setDescription(
            ""
        );
    }

}

void tensorFieldStorageFn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
//     /*
//     * This pass moves some fields of layers to be stored in a Tensor.
//     * This is due to some limitations on the MvTensor & Blob Structures.
//     */

//     using namespace mv;

//     ControlModel cm(model);
//     DataModel dm(model);
//     OpModel om(model);

//     std::cout << "Pass Enabled " << std::endl;

//     for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
//     {

//         if (opIt->getOpType() == OpType::Add  || opIt->getOpType() == OpType::Subtract || opIt->getOpType() == OpType::Multiply)
//         {
//             // EltwiseOp e(opIt);
//             std::cout << "Pass Trigger " << std::endl;

//             dynamic_vector<mv::float_type> relu_fields;

//             relu_fields = dynamic_vector<mv::float_type> {
//                 0.0,  // Has Relu
//                 0.0,  // Relu Negative Slope
//                 0.0   // Relu Positive Slope
//             };
//             std::cout << "Pass Trigger 2" << std::endl;

//             std::string reluTensorName = opIt->getName() + "_reluTensor";
//             auto reluTensor = dm.defineTensor(reluTensorName, 3, mv::DType::Float, mv::Order::RowMajor, relu_fields);

//             std::cout << "Pass Trigger 3" << std::endl;
//             opIt->prepareForSerialization(reluTensor);
//         }
//     }

}