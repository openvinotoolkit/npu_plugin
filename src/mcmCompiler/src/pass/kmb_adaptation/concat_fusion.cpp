#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

typedef std::pair<std::string, std::string> pairs;
static void fuseConcatsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(FuseConcats)
            .setFunc(fuseConcatsFcn)
            .setDescription(
                "The idea of this pass is the following: If we have a concat leading to another concat, and there is\
                only one/unique connection/flow between them, then these concats can be fused. There is no reason for the\
                intermediate concat/tensor.");
    }
}

bool haveMatchingDimensions(mv::Data::OpListIterator op1, mv::Data::OpListIterator op2)
{
    bool matching_width_dimension = 
        (op1->getInputTensor(0UL)->getShape()[mv::IO_WIDTH_DIMENSION] == op2->getInputTensor(0UL)->getShape()[mv::IO_WIDTH_DIMENSION]);
    bool matching_hight_dimension = 
        (op1->getInputTensor(0UL)->getShape()[mv::IO_HEIGHT_DIMENSION] == op2->getInputTensor(0UL)->getShape()[mv::IO_HEIGHT_DIMENSION]);
    return matching_width_dimension && matching_hight_dimension;
}

void locateConcatPairs(mv::ComputationModel& model, std::vector<pairs> &acceptableFusedConcats)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto operations = om.topologicalSort();

    for(auto opIt = operations.begin(); opIt != operations.end(); ++opIt)
    {
        auto op = *opIt;
        if(op->getOpType() != "ImplicitConcat")
            continue;

        auto nextOps = findSinkLayers(dm, op->getOutputTensor(0));
        for (auto nextOp : nextOps)
        {
            if (nextOp->getOpType() == "ImplicitConcat")
            //NOTE: the condition about the same axis and the number of next concat equal 1 might not be mandatory,
            // but not the scope of this pr
                if (nextOp->get<std::string>("axis") == "C" && op->get<std::string>("axis") == "C" &&
                    nextOps.size() == 1)
                    acceptableFusedConcats.push_back({op->getName(), nextOp->getName()});
        }
    }
}

void storeConcatChildFrequency(const std::vector<pairs> &acceptableFusedConcats,
                                std::vector <std::string> &concatChilds,
                                std::map <std::string, std::size_t> &concatChildApperarences)
{
    std::set<std::string> frequencyConcats;

    for (auto concatPair : acceptableFusedConcats)
    {
        concatChilds.push_back(concatPair.second);
        frequencyConcats.insert(concatPair.second);
    }

    for (auto concatChild : frequencyConcats)
        for (auto concatPair : acceptableFusedConcats)
            if (concatPair.second == concatChild)
                concatChildApperarences[concatChild]++;

}

void findNextConcatChildOps(mv::ComputationModel& model, mv::Data::OpListIterator &concatChild,
    std::vector<mv::Data::OpListIterator> &nextOps, std::map <std::string, std::size_t> &nextOpSlotKeeper)
{
    mv::DataModel dm(model);
    nextOps = findSinkLayers(dm, concatChild->getOutputTensor(0));

    for (auto nextOp : nextOps)
    {
        for (std::size_t idx = 0; idx < nextOp->getInputTensor().size(); idx++)
        {
            if (nextOp->getInputTensor()[idx]->getName() == concatChild->getOutputTensor(0)->getName())
                nextOpSlotKeeper[nextOp->getName()] = idx;
        }
    }
}

//NOTE: the idea of this function is that it will populate the map with the names and the slots of the parent concats, ops,
//which lead to a childConcat
void populateTheSlotKeeper(mv::ComputationModel& model, std::map <std::string, std::size_t> &slotKeeper, mv::Data::OpListIterator &concatChild)
{
    mv::OpModel om(model);

    //keep the previous slots of the child inputs
    for (std::size_t idx = 0; idx < concatChild->getInputTensor().size(); idx++)
        slotKeeper[concatChild->getInputTensor()[idx]->getName()] = idx;

    std::size_t newConcatIdx = 0;
    bool foundConcat = false;
    for (std::size_t idx = 0; idx < concatChild->getInputTensor().size(); idx++)
    {
        auto previousOp = om.getSourceOp(concatChild->getInputTensor()[idx]);
        if (previousOp->getOpType() == "ImplicitConcat" && haveMatchingDimensions(concatChild, previousOp)) //NOTE: belongs To first, normally!!
        {
            foundConcat = true;
            slotKeeper.erase(concatChild->getInputTensor()[idx]->getName());

            for (std::size_t parentIdx = 0; parentIdx < previousOp->getInputTensor().size();
                parentIdx++)
            {
                auto parentInputTensor = previousOp->getInputTensor()[parentIdx];
                slotKeeper[parentInputTensor->getName()] = parentIdx + newConcatIdx;
            }
            newConcatIdx +=  previousOp->getInputTensor().size();
        }
        else
        {
            std::size_t unpopulatedInputs = 0;
            for (auto inputTensor : previousOp->getInputTensor())
                if (!inputTensor->isPopulated())
                    unpopulatedInputs++;
            if (foundConcat)
                slotKeeper[concatChild->getInputTensor()[idx]->getName()] = newConcatIdx;
            newConcatIdx += unpopulatedInputs;
        }
    }
}

void sortInputTensors(mv::ComputationModel& model, std::map <std::string, std::size_t> &slotKeeper,
                        std::vector<mv::Data::TensorIterator> &sortedInputTensors)
{
    mv::DataModel dm(model);
    for (std::size_t idx = 0 ; idx < slotKeeper.size(); idx++)
    {
        for (auto slotCase : slotKeeper)
            if (idx == slotCase.second)
                sortedInputTensors.push_back(dm.getTensor(slotCase.first));
    }

}

void placeNewConcatInTheOpModel(mv::ComputationModel& model, mv::Data::OpListIterator &concatChild,
    std::vector<mv::Data::TensorIterator> &sortedInputTensors, std::vector<pairs> &acceptableFusedConcats,
    std::vector<mv::Data::OpListIterator> &nextOps, std::map <std::string, std::size_t> &nextOpSlotKeeper)
{
    mv::OpModel om(model);

    auto concat = om.implicitConcat("Fusedconcat_" + concatChild->getName(), sortedInputTensors, "C");
    concat->setDType(concatChild->getOutputTensor(0)->getDType());
    concat->setQuantParams(concatChild->getOutputTensor()[0]->get<mv::QuantizationParams>("quantParams"));
    concat->set<std::string>("splitStrategy", concatChild->getOutputTensor()[0]->get<std::string>("splitStrategy"));
    concat->set<mv::Tensor::MemoryLocation>("Location", concatChild->getOutputTensor()[0]->get<mv::Tensor::MemoryLocation>("Location"));

    om.getSourceOp(concat)->set<std::string>("splitStrategy", "Clustering");
    om.getSourceOp(concat)->set<unsigned>("opId", concatChild->get<unsigned>("opId"));


    for (auto pairIt : acceptableFusedConcats)
    {
        if (pairIt.second == concatChild->getName())
        {
            om.removeOp(om.getOp(pairIt.first));
        }
    }
    om.removeOp(concatChild);

    //NOTE: The sinkOps of the concats could need to be connected on a different slot rather than 0!!
    for (std::size_t idx = 0; idx < nextOps.size(); idx++)
    {
        om.defineFlow(concat, nextOps[idx], nextOpSlotKeeper[nextOps[idx]->getName()]);
        nextOps[idx]->setInputTensor(concat, nextOpSlotKeeper[nextOps[idx]->getName()], false);
    }

}

void fuseConcatsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    //NOTE: find the concat pairs that can be fused
    std::vector<pairs> acceptableFusedConcats;
    locateConcatPairs(model, acceptableFusedConcats);

    //NOTE: keeps how many times the child concat appear
    std::map <std::string, std::size_t> concatChildApperarences;
    std::vector <std::string> concatChilds;
    storeConcatChildFrequency(acceptableFusedConcats, concatChilds, concatChildApperarences);

    for (auto concatPair : acceptableFusedConcats)
    {
        auto concatChild = om.getOp(concatPair.second);

        //NOTE: update the appropriate variables according to the concat child
        concatChildApperarences[concatPair.second]--;

        if (concatChildApperarences[concatPair.second] == 0)
        {
            //NOTE: keep the next ops after the child concat so you can connect later
            std::vector<mv::Data::OpListIterator> nextOps;
            std::map <std::string, std::size_t> nextOpSlotKeeper;
            findNextConcatChildOps(model, concatChild, nextOps, nextOpSlotKeeper);

            std::map <std::string, std::size_t> slotKeeper;
            populateTheSlotKeeper(model, slotKeeper, concatChild);

            //NOTE: Sort the input Tensors of the new Concat
            std::vector<mv::Data::TensorIterator> sortedInputTensors;
            sortInputTensors(model, slotKeeper, sortedInputTensors);

            placeNewConcatInTheOpModel(model, concatChild, sortedInputTensors, acceptableFusedConcats, nextOps, nextOpSlotKeeper);
        }
    }
}