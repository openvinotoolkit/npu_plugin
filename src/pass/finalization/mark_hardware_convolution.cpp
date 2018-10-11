#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "mcm/utils/custom_math.hpp"
#include <math.h>

static void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkHardwareConvolution)
        .setFunc(markHardwareConvolution)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass marks the convolutions that can be executed in NCE"
        );

        MV_REGISTER_PASS(ScaleFission)
        .setFunc(scaleFissionFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Adds scales around HW ops to utilize more bits of fixed-point number representation in MAC HW units"
        );
    }
}

//NOTE: This should not be done in such hardcoded way.
void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    //int amount_marked = 0;
    //int mark_limit = 3;

    bool disableHardware = false;
    if (compDesc.hasKey("MarkHardwareConvolution"))
        if (compDesc["MarkHardwareConvolution"].hasKey("disableHardware"))
            if (compDesc["MarkHardwareConvolution"]["disableHardware"].get<bool>())
                disableHardware = true;

    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!disableHardware)
        {
            if(!opIterator->isHardwarizeable(compDesc))// || amount_marked >= mark_limit)
            {
                om.addAttr(opIterator, "NCE1_Compatible", (int)0);
                continue;
            }

            om.addAttr(opIterator, "NCE1_Compatible", (int)1);
            om.addAttr(opIterator, "NCE1_AssignedCMX", (int)0);
            //++amount_marked;
        }
        else
            om.addAttr(opIterator, "NCE1_Compatible", (int)0);
    }
}

void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    std::string const PASS_NAME = "ScaleFission";
    std::string const FACTOR_KEY = "scalefactors";
    std::string opName = "";

    double upNum = 1.0;

    if (!compDesc.hasKey("pass"))
    {
        return ;
    }
    if (!compDesc["pass"].hasKey(PASS_NAME))
    {
        return ;
    }
    if (!compDesc["pass"][PASS_NAME].hasKey(FACTOR_KEY))
    {
        return ;
    }

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        opName =opIt->getName();
        if ((opIt->getOpType() == OpType::Conv2D)&&(opIt->hasAttr("NCE1_Compatible")))
        {
            if (opIt->get<int>("NCE1_Compatible") == 1)
            {
                if (compDesc["pass"][PASS_NAME][FACTOR_KEY].hasKey(opName))
                {
                    upNum = compDesc["pass"][PASS_NAME][FACTOR_KEY][opName].get<double>();

                    std::vector<double> scaleUpWData = mv::utils::generateSequence<double>(opIt->getInputTensor(1)->getShape().totalSize(), upNum, 0.0f);
                    std::vector<double> scaleDnData = mv::utils::generateSequence<double>(opIt->getOutputTensor(0)->getShape().totalSize(), (1.0f/upNum), 0.0f);

                    // scale (up) inputs by multiplying weights and bias
                    std::string scaleUpWTensorName = opName + "_scale_in";
                    auto scaleUpWeights = dm.defineTensor(scaleUpWTensorName, opIt->getInputTensor(1)->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleUpWData);
                    opIt->getInputTensor(1)->multiply(*scaleUpWeights);

                    if (opIt->hasAttr("bias"))
                    {
                        auto biasTensor = dm.findTensor(opIt->get<std::string>("bias"));
                        std::vector<double> scaleUpBData = mv::utils::generateSequence<double>(biasTensor->getShape().totalSize(), upNum, 0.0f);
                        std::string scaleUpBTensorName = opName + "_scale_bias";
                        auto scaleUpBias = dm.defineTensor(scaleUpBTensorName, biasTensor->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleUpBData);
                        biasTensor->multiply(*scaleUpBias);
                    }

                    // scale (down) output by adding HWscale attributes to conv
                    std::string scaleTensorName = opName + "_scale";
                    auto scaleTensor = dm.defineTensor(scaleTensorName, opIt->getOutputTensor(0)->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, scaleDnData);
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
