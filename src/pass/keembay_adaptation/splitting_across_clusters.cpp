#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void splittingAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void getWorkGen(const mv::Data::TensorIterator& tensor, const int nClusters);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SplittingAcrossClusters)
        .setFunc(splittingAcrossClusters)
        .setDescription(
            "Computing Splitting across clusters"
        );
    }
}

void splittingAcrossClusters(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    std::vector <mv::Data::TensorIterator> tensors;
    auto globalParams = model.getGlobalConfigParams();
    auto numClusters = globalParams->get("Number_of_Clusters");

    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        std::string opType = layer->getOpType();
        if (opType == "DPUTask")
        {
            auto outputTensor = layer->getOutputTensor(0);
            tensors.push_back(outputTensor);
            unsigned n = layer->inputSlots();
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = layer->getInputTensor(i);
                tensors.push_back(inputTensor);
            }
        }
     }

    for (std::vector<mv::Data::TensorIterator>::iterator tensor = tensors.begin(); tensor != tensors.end(); ++tensor)
    {
        getWorkGen(*tensor, numClusters);
    }
    return;
}

void getWorkGen(const mv::Data::TensorIterator& tensor, const int nClusters)
{
    switch (tensor->get<std::string>("splitStrategy"))
    {
        case "SplitOverH":
        if(!inputTensor->isPopulated())
        {
            int nWorkloads = nClusters;
        }
        else
        {
            int nWorkloads = 1;
        }
        break;
        default;
    }
    return;
}
