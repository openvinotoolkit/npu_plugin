#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/target/keembay/workload_struct.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"

#define UNUSED(expr) do {(void)(expr);} while (0)
static const mv::DPUModeList TENSOR_MPE = {{1,1}};


static void splittingAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void subTensorsGen(const std::vector<mv::Data::TensorIterator> &tensors, const int nClusters, const mv::pass::PassEntry& pass);

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

void splittingAcrossClusters(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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

    subTensorsGen(tensors, numClusters, pass);
    return;
}

void subTensorsGen(const std::vector <mv::Data::TensorIterator>& tensors, const int nClusters, const mv::pass::PassEntry& pass)
{
    for (auto tensor = tensors.begin(); tensor != tensors.end(); ++tensor)
    {
        if ((*tensor)->get<std::string>("splitStrategy") == "SplitOverH")
        {
            mv::Workloads Tensor((*tensor)->getName(), (*tensor)->getShape());
            int success;
            UNUSED(success);
            std::vector<mv::Workload> subTensors;
            if(!(*tensor)->isPopulated())
            {
                int nWorkloads = nClusters;
                success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE, nWorkloads, true, false, true, mv::WorkloadSplitMode::HW, pass);
                subTensors = Tensor.getWorkloads();
            }
            else
            {
                success = Tensor.partitionTensorWithRectangleHeuristic(TENSOR_MPE, 1, true, false, true, mv::WorkloadSplitMode::HW, pass);
                subTensors = Tensor.getWorkloads();
                std::vector<mv::Workload> newSubTensors;
                for (int i = 0; i < nClusters; i++)
                {
                    for (unsigned int j =0; j < subTensors.size(); j++)
                        newSubTensors.push_back(subTensors[j]);
                }
                subTensors = newSubTensors;
            }
            (*tensor)->splitAcrossClusters(subTensors, true, false);
        }
    }
    return;
}
