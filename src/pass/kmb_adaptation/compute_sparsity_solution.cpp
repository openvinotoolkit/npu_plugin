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
    auto convOps = om.getOps("Conv");

    for (auto& convOp : convOps)
    {
        auto inputTensor = convOp->getInputTensor(0);
        auto inputTensorMemoryLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
        //if the input Tensor is in ddr, it means that the previous op outputed in cmx and for
        //memory requirements a travel to ddr is taking place so the previous op was streamed
        if (inputTensorMemoryLocation == mv::Tensor::MemoryLocation("DDR"))
        {

            //for now we are going to handle only the case that we have an op flaot16
            if (convOp->hasAttr("floatPrecision") && convOp->get<bool>("floatPrecision"))
            {
                convOp->set<bool>("activationSparsityCompilerSolving", true);
                convOp->set<bool>("inputActivationSparsity", true);

            }

        }
        if (convOp->hasAttr("DilatedSubConv") && convOp->get<bool>("DilatedSubConv"))
        {
            if (inputTensorMemoryLocation == mv::Tensor::MemoryLocation("NNCMX"))
            {
                convOp->set<bool>("activationSparsityCompilerSolvingForDilatedConv", true);
                convOp->set<bool>("inputActivationSparsityForDilatedConv", true);
            }
        }

    }
}
