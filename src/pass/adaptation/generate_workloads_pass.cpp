#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include <math.h>

static void generateWorkloadsFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWorkloads)
            .setFunc(generateWorkloadsFcn)
            .setGenre(PassGenre::Adaptation)
            .setDescription(
                "This pass generates workloads");
    }
}

std::set<int> getSplitsFromRange(int maxSplitRange, int maxSplits = 50)
{
    std::set<int> splits;

    if((maxSplitRange < maxSplits) && (maxSplitRange >1)) 
    {
        splits.insert(maxSplitRange);
        do 
        {
            maxSplitRange = maxSplitRange >> 1;
            splits.insert(maxSplitRange);
        } 
        while ((maxSplitRange >> 1) > 1);
    }
    return splits;
}

std::set<int> getNWorkloads(std::vector<mv::Data::TensorIterator> tensor, int nDPUxClusterS)
{
    std::cout << "Test getNWorkloads" << std::endl;
    std::cout << "Tensor shape is " << tensor[0]->getShape().toString() << std::endl;

    //maxSplitsXY
    auto xDim = tensor[0]->get<mv::Shape>("shape")[0];
    auto yDim = tensor[0]->get<mv::Shape>("shape")[1];
    auto maxSplitsXY = ceil(xDim/4) * ceil(yDim/4);

    std::cout << "maxSplitsXY is " << maxSplitsXY << std::endl;

    //maxSplitsZ
    auto maxSplitsZ = ceil(tensor[0]->get<mv::Shape>("shape")[2]/16);

    std::cout << "maxSplitsZ is " << maxSplitsZ << std::endl;

    //Pool of possible splits
    std::set<int> XYTileSplits;
    std::set<int> ZTileSplits;
    std::set<int> splitPool;

    XYTileSplits = getSplitsFromRange(maxSplitsXY);
    ZTileSplits = getSplitsFromRange(maxSplitsZ);

    std::set_union(std::begin(XYTileSplits), std::end(XYTileSplits),
               std::begin(ZTileSplits), std::end(ZTileSplits),                  
               std::inserter(splitPool, std::begin(splitPool)));
    
    return splitPool;
}

void generateWorkloadsFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{
    using namespace mv;

    mv::OpModel om(model);

    int nDPU = 3;
    int nClusters = 1;
    int nDPUxCluster = nDPU/nClusters;
    std::set<int> workloadsList;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            std::cout << "Found DPUTask" << std::endl;

            auto outputTensor = opIt->getOutputTensor();

            workloadsList = getNWorkloads(outputTensor, nDPUxCluster);
        }
    }

    //Already began to calculate a pool of workloads as per PoC compiler but for ww09 
    //Forcing number of workloads to be nDPU/nCluster round to nearest even number
    auto nWorkloads = round(nDPUxCluster/2)*2; 

    std::cout << "Number of workloads is " << nWorkloads << std::endl;

    //Workload class
    Workloads workloads;

    mv::Workload w1;   
    w1.MinX = 0;
    w1.MaxX = 15;
    w1.MinY = 0;
    w1.MaxY = 3;
    w1.MinZ = 0;
    w1.MaxZ = 15;

    workloads.getWorkloads().push_back(w1);

    

    std::cout << "Exiting Workload Generation Pass " << std::endl;
}
