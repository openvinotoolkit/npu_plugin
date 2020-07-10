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
        // Trim out channel major convolutions
        auto new_end = std::remove_if(opsMap.at("Conv").begin(), opsMap.at("Conv").end(),
                    [](const mv::Data::OpListIterator op)
                    {return op->getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16;});
        opsMap.at("Conv").erase(new_end, opsMap.at("Conv").end());
    }

    auto globalParams = model.getGlobalConfigParams();
    auto referenceDevice = globalParams->get<std::string>("referenceDevice");

    // A0 FP16 input sparsity requirements
    for (auto opList : opsMap) {
        for (auto op : opList.second) {

            bool solvedByMixedConversion = op->hasAttr("placeConversionToFloat") &&
                op->get<bool>("placeConversionToFloat") &&
                op->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") ==
                mv::Tensor::MemoryLocation("NNCMX");

            if (referenceDevice == "A0" &&
                op->hasAttr("floatPrecision") &&
                op->get<bool>("floatPrecision") &&
                (!op->hasAttr("inputActivationSparsity") ||
                !op->get<bool>("inputActivationSparsity")) &&
                !solvedByMixedConversion)
            {
                op->set<bool>("activationSparsityCompilerSolving", true);
            }
        }
    }

    // A0 ZM Conv SOH input sparsity requirements
    for (auto convOp : opsMap["Conv"])
    {
        if (referenceDevice == "A0" &&
            convOp->hasAttr("splitStrategy") &&
            convOp->get<std::string>("splitStrategy") == "SplitOverH" &&
            (convOp->getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] > 1 ||
            convOp->getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] > 1) &&
            (!convOp->hasAttr("inputActivationSparsity") ||
            !convOp->get<bool>("inputActivationSparsity")))
        {
            convOp->set<bool>("activationSparsityCompilerSolving", true);
        }
    }

    // Dilated convolution preocessing optimization requiring sparsity
    for (auto convOp : opsMap["Conv"])
    {
        if (convOp->hasAttr("DilatedSubConv") && convOp->get<bool>("DilatedSubConv") &&
            !(convOp->hasAttr("slicedInput3DDMA") && convOp->get<bool>("slicedInput3DDMA")))
        {
            convOp->set<bool>("activationSparsityCompilerSolvingForDilatedConv", true);
            convOp->set<bool>("inputActivationSparsityForDilatedConv", true);
        }
    }
}
