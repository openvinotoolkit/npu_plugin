#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

static void preprocessForPWL(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(PreprocessForPWL)
        .setFunc(preprocessForPWL)
        .setDescription(
            "Preprocess appropriately the operations that have leaky Relu as a post Op for the PWL approach:\
            You need to mark the first guy cause it is going to load the registers, the rest need to take FLEXARB."
        );
    }
}

void preprocessForPWL(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    bool PWLUsage = globalParams->hasAttr("PWLUsage") ? globalParams->get<bool>("PWLUsage") : false;
    if (PWLUsage)
    {
        //NOTE: find the first convolution that has lrelu as postOp
        bool foundFirstConv = false;
        auto sortedOps = om.topologicalSort();
        mv::Data::OpListIterator firstConv;
        for (auto opIterator: sortedOps)
        {
            if (opIterator->getOpType() == "Conv" && opIterator->hasAttr("postOpTypes"))
            {
                auto postOpTypes = opIterator->get<std::vector<std::string>>("postOpTypes");
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "LeakyRelu") != postOpTypes.end())
                {
                    foundFirstConv = true;
                    firstConv = opIterator;
                    break;
                }
            }
        }
        //NOTE: opIterator contains the first conv with lrelu if no conv with lrelu just go to next pass
        if (foundFirstConv)
        {
            firstConv->set<bool>("firstConvWithLRelu", true);
            auto convOps = om.getOps("Conv");
            for (auto& convOp : convOps)
            {
                if (convOp->hasAttr("postOpTypes"))
                {
                    auto postOpTypes = convOp->get<std::vector<std::string>>("postOpTypes");
                    auto itr = std::find(postOpTypes.begin(), postOpTypes.end(), "LeakyRelu");
                    if (itr != postOpTypes.end())
                    {
                        replace(postOpTypes.begin(), postOpTypes.end(), +"LeakyRelu", +"FLEXARB");
                        convOp->set<std::vector<std::string>>("postOpTypes", postOpTypes);
                    }
                }
            }
        }
    }
    return;
}
