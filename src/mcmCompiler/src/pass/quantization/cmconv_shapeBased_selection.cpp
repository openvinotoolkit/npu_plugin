#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void disableCMconvOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(SelectiveCMconv)
        .setFunc(disableCMconvOpFcn)
        .setDescription(
            "Disable CMConv for networks based on topology"
        );
    }

}

//CM Conv temporary fix. SSDs, VGG16, Yolov2 and Yolov3 failing or don't have support for CM Conv yet
//So disabling CM Conv for these networks based on the network identiers
void disableCMconvOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    auto globalParams = model.getGlobalConfigParams();
    bool enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
    if (enableChannelMajorConv)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
        mv::OpModel om(model);
        mv::DataModel dm(model);

        //Adding FQ check to differentiate between OV and Tflite models and enable/disable CMConv accordingly

        bool hasFQ = false;
        auto fqOps = om.getOps("FakeQuantize");
        auto convOps = om.getOps("Conv");
        auto inputOp = om.opBegin();

        if(!fqOps.empty())
            hasFQ = true;
        else
            return;

        if(hasFQ)
        {
            for (auto& convOp : convOps)
            {
                if ((convOp->getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 512) ||
                (convOp->getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 300))
                {
                   globalParams->erase("enable_channel_major_conv");
                   globalParams->set("enable_channel_major_conv", mv::Attribute(bool(false)));
                }
            }

            auto sinkOperators = findSinkLayers(dm, inputOp->getOutputTensor(0));
            if (sinkOperators[0]->getOpType() != "Scale")
            {
                auto nextSinkOperators = findSinkLayers(dm, sinkOperators[0]->getOutputTensor(0));
                if (nextSinkOperators[0]->getOpType() == "Conv"){
                   globalParams->erase("enable_channel_major_conv");
                   globalParams->set("enable_channel_major_conv", mv::Attribute(bool(false)));
                }
            }
        }
    }
}
