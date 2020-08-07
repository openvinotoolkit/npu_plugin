#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <regex>

static void computeSparsitySolutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeSparsitySolution)
        .setFunc(computeSparsitySolutionFcn)
        .setDescription(
            "This pass predicts from who the unpopulated sparsity will be solved runtime/compiler."
        );
    }
}

void propagateRealSparsityLoss(mv::OpModel& om, mv::Data::OpListIterator op)
{
    // Give up on proper runtime generated output sparsity
    op->set<bool>("outputActivationSparsity", false);
    for(auto flow = op.leftmostOutput(); flow != om.flowEnd(); ++flow)
    {
        // Proper sparsity does not propagate through implicit ops
        // Currently unimplemented
        auto childOp = flow.sink();
        auto isDilatedConv =
            childOp->hasAttr("activationSparsityCompilerSolvingForDilatedConv") &&
            childOp->get<bool>("activationSparsityCompilerSolvingForDilatedConv");

        if (childOp->hasAttr("inputActivationSparsity") &&
            childOp->get<bool>("inputActivationSparsity") &&
            !isDilatedConv)
            childOp->set<bool>("activationSparsityCompilerSolving", true);
    }
}

void computeSparsitySolutionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    //IDU OF z-major Conv supports sparsity, so take all the input tensors of convs,
    //see where they are located, if they are on DDR and they need sparsity mark them
    //cause their sparsity is going to be solved from populated pass sparse. even if they
    //are unpopulated

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto opsMap = om.getOpsOfTypes({"Conv", "Eltwise"});

    if (model.getGlobalConfigParams()->get<bool>("enable_channel_major_conv"))
    {
        auto convs = om.getOps("Conv");

        if(opsMap["Conv"].size())
        {
            // Trim out channel major convolutions
            auto new_end = std::remove_if(opsMap.at("Conv").begin(), opsMap.at("Conv").end(),
                        [](const mv::Data::OpListIterator op)
                        {return op->getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16;});
            opsMap.at("Conv").erase(new_end, opsMap.at("Conv").end());
        }
    }

    for (auto convOp : opsMap["Conv"])
    {
        // Dilated convolution HW processing optimization, requires custom
        // sparsity generation which can't be serviced by runtime.
        if (convOp->hasAttr("DilatedSubConv") && convOp->get<bool>("DilatedSubConv") &&
            !(convOp->hasAttr("slicedInput3DDMA") && convOp->get<bool>("slicedInput3DDMA"))) {
            convOp->set<bool>("activationSparsityCompilerSolvingForDilatedConv", true);
            propagateRealSparsityLoss(om, om.getSourceOp(convOp->getInputTensor(0)));
        }
    }

    // Decide wheter convolution sparsity is runtime or compiler solved
    for (auto convOp : opsMap["Conv"])
    {
        auto parentOp = om.getSourceOp(convOp->getInputTensor(0));
        if((!parentOp->hasAttr("outputActivationSparsity") ||
            !parentOp->get<bool>("outputActivationSparsity")) &&
            (convOp->hasAttr("inputActivationSparsity") &&
            convOp->get<bool>("inputActivationSparsity")))
            propagateRealSparsityLoss(om, parentOp);
    }

    // Decide wheter eltwise sparsity is runtime or compiler solved
    for (auto eltwiseOp : opsMap["Eltwise"])
    {
        auto parentOutputSparsity = true;
        for(auto parentOp = eltwiseOp.leftmostParent();
            parentOp != om.opEnd(); ++parentOp)
        {
            parentOutputSparsity = parentOutputSparsity &&
                parentOp->hasAttr("outputActivationSparsity") &&
                parentOp->get<bool>("outputActivationSparsity");
        }
        if(!parentOutputSparsity &&
            (eltwiseOp->hasAttr("inputActivationSparsity") &&
            eltwiseOp->get<bool>("inputActivationSparsity"))) {
            for(auto parentOp = eltwiseOp.leftmostParent();
                parentOp != om.opEnd(); ++parentOp)
                propagateRealSparsityLoss(om, parentOp);
            eltwiseOp->set<bool>("activationSparsityCompilerSolving", true);
        }
    }
}
