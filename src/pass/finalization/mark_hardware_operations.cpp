#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "mcm/utils/custom_math.hpp"
#include <math.h>

static void markHardwareOperations(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);
static void scaleFissionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkHardwareOperations)
        .setFunc(markHardwareOperations)
        .setDescription(
            "This pass marks the operations that can be executed in NCE."
        );

        MV_REGISTER_PASS(ScaleFission)
        .setFunc(scaleFissionFcn)
        .setDescription(
            "Adds scales around HW ops to utilize more bits of fixed-point number representation in MAC HW units"
        );
    }
}

//NOTE: This should not be done in such hardcoded way.
void markHardwareOperations(const mv::pass::PassEntry &, mv::ComputationModel& model, mv::TargetDescriptor& targetDescriptor, mv::Element& passDesc, mv::json::Object &)
{

    //int amount_marked = 0;
    //int mark_limit = 3;

    bool disableHardware = false;
    if (passDesc.hasAttr("disableHardware"))
        if (passDesc.get<bool>("disableHardware"))
            disableHardware = true;

    mv::OpModel om(model);
    mv::Target target = targetDescriptor.getTarget();

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        // std::cout << " "opIterator->getName() << std::endl;
        if (!disableHardware)
        {

            if (opIterator->getOpType() == "Conv")
            {
                auto padding = opIterator->get<std::array<unsigned short, 4>>("padding");
                auto stride = opIterator->get<std::array<unsigned short, 2>>("stride");

                auto input = opIterator->getInputTensor(0);
                auto inputShape = input->getShape();
                auto weights = opIterator->getInputTensor(1);
                auto weightsShape = weights->getShape();

                if (target == mv::Target::ma2480)
                {
                    // Check for supported padding
                    if ((padding[0] != 0 && padding[0] != weightsShape[0]/2) || (padding[2] != 0 && padding[2] != weightsShape[1]/2))
                    {
                        om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                        continue;
                    }

                    // Check for supported kernel sizes
                    if(weightsShape[0] > 15 || weightsShape[1] > 15)
                    {
                        om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                        continue;
                    }

                    // Check for supported strides
                    if(stride[0] > 8 || stride[1] > 8)
                    {
                        om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                        continue;
                    }
                }
                else //ma2490
                {
                    // Check for supported kernel sizes
                    if(weightsShape[0] > 11 || weightsShape[1] > 11)
                    {
                        om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                        continue;
                    }

                    // Check for supported strides
                    if(stride[0] > 8 || stride[1] > 8)
                    {
                        om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                        continue;
                    }
                }
                om.addAttr(opIterator, "NCE1_Compatible", (int)1);
                om.addAttr(opIterator, "NCE1_AssignedCMX", (int)0);

            }
            else if (opIterator->getOpType() == "MaxPool")
            {
                om.addAttr(opIterator, "NCE1_Compatible", (int)1);
                om.addAttr(opIterator, "NCE1_AssignedCMX", (int)0);
            }
            else if (opIterator->getOpType() == "AveragePool")
            {
                om.addAttr(opIterator, "NCE1_Compatible", (int)1);
                om.addAttr(opIterator, "NCE1_AssignedCMX", (int)0);
            }
            else
            {
                om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                continue;
            }

        }
        else
            om.addAttr(opIterator, "NCE1_Compatible", (int)0);
    }
}

void scaleFissionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    std::string const FACTOR_KEY = "scalefactors";
    std::string opName = "";

    double upNum = 1.0;

    mv::Element& factorKeyElem = passDesc.get<mv::Element>(FACTOR_KEY);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        opName =opIt->getName();
        if (opIt->getOpType() == "Conv" && opIt->hasAttr("NCE1_Compatible"))
        {
            if (opIt->get<int>("NCE1_Compatible") == 1)
            {
                if (factorKeyElem.hasAttr(opName))
                {
                    upNum = factorKeyElem.get<double>(opName);

                    std::vector<double> scaleUpWData = mv::utils::generateSequence<double>(opIt->getInputTensor(1)->getShape().totalSize(), upNum, 0.0f);
                    std::vector<double> scaleDnData = mv::utils::generateSequence<double>(opIt->getOutputTensor(0)->getShape().totalSize(), (1.0f/upNum), 0.0f);

                    // scale (up) inputs by multiplying weights and bias
                    std::string scaleUpWTensorName = opName + "_scale_in";
                    auto scaleUpWeights = dm.defineTensor(scaleUpWTensorName, opIt->getInputTensor(1)->getShape(),DType("Float16"), mv::Order("HWC"), scaleUpWData);
                    opIt->getInputTensor(1)->multiply(*scaleUpWeights);

                    if (opIt->hasAttr("bias"))
                    {
                        auto biasTensor = dm.getTensor(opIt->get<std::string>("bias"));
                        std::vector<double> scaleUpBData = mv::utils::generateSequence<double>(biasTensor->getShape().totalSize(), upNum, 0.0f);
                        std::string scaleUpBTensorName = opName + "_scale_bias";
                        auto scaleUpBias = dm.defineTensor(scaleUpBTensorName, biasTensor->getShape(), DType("Float16"), mv::Order("HWC"), scaleUpBData);
                        biasTensor->multiply(*scaleUpBias);
                    }

                    // scale (down) output by adding HWscale attributes to conv
                    std::string scaleTensorName = opName + "_scale";
                    auto scaleTensor = dm.defineTensor(scaleTensorName, opIt->getOutputTensor(0)->getShape(), DType("Float16"), mv::Order("HWC"), scaleDnData);
                    Attribute scaleAttr(scaleTensor->getName());
                    om.addAttr(opIt, "scale", scaleAttr);
                }
                else
                {
                    upNum = 1.0;
                }
            }  // end HW conv
        }  // end conv
    }  // end op loop
}
