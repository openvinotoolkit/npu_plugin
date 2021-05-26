#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

typedef std::pair<std::string, std::string> pairs;
static void fuseImplicitsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void fuseConcatsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void fuseCropSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void fuseCropStridedSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void fuseSliceSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(FuseImplicits)
            .setFunc(fuseImplicitsFcn)
            .setDescription(
                "Fuse back-to-back implicit ops (e.g., Concat-to-Concat, Crop-to-StridedSlice, Slice-to-Slice");
    }
}

void fuseImplicitsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    fuseConcatsFcn(pass, model);
    fuseCropSliceFcn(pass, model);
    fuseCropStridedSliceFcn(pass, model);
    fuseSliceSliceFcn(pass, model);
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
        if (previousOp->getOpType() == "ImplicitConcat") //NOTE: belongs To first, normally!!
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

// If we have a concat leading to another concat, and there is only one/unique connection/flow
// between them, then these concats can be fused. There is no reason for the intermediate concat/tensor
void fuseConcatsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model)
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

// Fuse Crop & Slice
void fuseCropSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto cropOps = om.getOps("Crop");
    for (auto cropOp : cropOps)
    {
        auto crop = cropOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto parentOutputTensor = cropOp->getInputTensor(mv::IO_TENSOR_INPUT);
        auto parentOp = om.getSourceOp(parentOutputTensor);
        auto childOp = cropOp.leftmostOutput().sink();

        // Skip if child isn't Slice
        if (childOp->getOpType() != "Slice")
            continue;

        pass.log(mv::Logger::MessageType::Debug, "Fuse Crop & Slice: " + cropOp->getName() + ", & " + childOp->getName());

        // Remove Crop
        linkNewOperationsRemove(parentOp, parentOutputTensor, om, cropOp);
    }
}

// Fuse Crop & StridedSlice
void fuseCropStridedSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto cropOps = om.getOps("Crop");
    for (auto cropOp : cropOps)
    {
        auto crop = cropOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto parentOutputTensor = cropOp->getInputTensor(mv::IO_TENSOR_INPUT);
        auto parentOp = om.getSourceOp(parentOutputTensor);
        auto childOp = cropOp.leftmostOutput().sink();

        // Skip if child isn't StridedSlice
        if (childOp->getOpType() != "UPATask")
            continue;
        if (!(childOp->hasAttr("taskOp") && childOp->get<std::string>("taskOp") == "StridedSlice"))
            continue;

        // Skip if strides not all 1
        auto strides = childOp->get<std::vector<unsigned>>("strides");
        if (!(strides[0] == 1 && strides[1] == 1 && strides[2] == 1 && strides[3] == 1))
            continue;

        pass.log(mv::Logger::MessageType::Debug, "Fuse Crop & StridedSlice: " + cropOp->getName() + ", & " + childOp->getName());

        // New slice_begin is StridedSlice's begin, since crop begin is all 0
        auto stridedSlice_begins = childOp->get<std::vector<unsigned>>("begins");
        auto slice_begin = mv::Shape({stridedSlice_begins[3], stridedSlice_begins[2], stridedSlice_begins[1], stridedSlice_begins[0]});

        // Calculate new slice_size
        auto stridedSlice_ends = childOp->get<std::vector<unsigned>>("ends");
        auto slice_end = mv::Shape({stridedSlice_ends[3], stridedSlice_ends[2], stridedSlice_ends[1], stridedSlice_ends[0]});
        auto slice_size = slice_end - slice_begin;

        // Remove Crop
        auto cropName = cropOp->getName();
        linkNewOperationsRemove(parentOp, parentOutputTensor, om, cropOp);

        // Add Slice
        auto stridedSlice = childOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
        auto dtype = stridedSlice->getDType();
        auto quantParams = stridedSlice->getQuantParams();
        auto splitStrategy = stridedSlice->get<std::string>("splitStrategy");
        auto slice = om.slice(cropName + "_fuseStridedSlice", parentOp->getOutputTensor(mv::IO_TENSOR_OUTPUT), slice_begin, slice_size);
        slice->setDType(dtype);
        slice->setQuantParams(quantParams);
        slice->set<std::string>("splitStrategy", splitStrategy);
        if(childOp->hasAttr("opId")) {
            unsigned opId = childOp->get<unsigned>("opId");
            slice->set<unsigned>("opId", opId);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);
        }

        // Replace StridedSlice with Slice
        linkNewOperationsReplacement(parentOp, slice, om, om.getSourceOp(stridedSlice));
    }
}

// Fuse back-to-back Slices
// This logic finds one slice driving multiple slices, e.g., a slice from asymmetric-stride Conv replacement & slices from StreamingOps
void fuseSliceSliceFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto sliceOps = om.getOps("Slice");
    for (auto sliceOp : sliceOps)
    {
        // Skip if not all children are slices
        auto all_children_are_slices = true;
        for (auto sinkFlow = sliceOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
            if (sinkFlow.sink()->getOpType() != "Slice")
                all_children_are_slices = false;

        if (all_children_are_slices == false)
            continue;

        // Update all child slice begins
        auto first_slice_begin = sliceOp->get<mv::Shape>("begin");
        for (auto sinkFlow = sliceOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            // Set new slice begin
            auto second_slice_begin = sinkFlow.sink()->get<mv::Shape>("begin");
            auto new_begin = first_slice_begin + second_slice_begin;
            sinkFlow.sink()->set<mv::Shape>("begin", new_begin);
            pass.log(mv::Logger::MessageType::Debug, "Adjust slice begin: " + sinkFlow.sink()->getName());
        }

        // Remove first slice
        auto parentOutputTensor = sliceOp->getInputTensor(mv::IO_TENSOR_INPUT);
        auto parentOp = om.getSourceOp(parentOutputTensor);
        pass.log(mv::Logger::MessageType::Debug, "Remove first slice: " + sliceOp->getName());
        linkNewOperationsRemove(parentOp, parentOutputTensor, om, sliceOp);
    }
}
