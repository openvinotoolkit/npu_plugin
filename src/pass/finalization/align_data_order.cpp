#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void addConversionLayersFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void alignConstOrderFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AddConversionLayers)
        .setFunc(addConversionLayersFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass adds conversion layers where needed."
        );

        MV_REGISTER_PASS(AlignConstOrder)
        .setFunc(alignConstOrderFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass adds conversion layers where needed."
        );

    }

}

void alignConstOrderFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    DataModel dm(model);
    OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == OpType::Constant)
        {

            if (opIt.childrenSize() > 1)
                throw OpError(*opIt, "Constant has more than 1 children, currently unsupported");

            // Constant is a parameter tensor for a hardware layer
            if (opIt.leftmostChild()->hasAttr("NCE1_Compatible"))
            {
                if (opIt.leftmostChild()->get<int>("NCE1_Compatible") == 1)
                {
                    //opIt->set<Order>("order", OrderType::RowMajorPlanar);
                    opIt.leftmostOutput()->getTensor()->setOrder(OrderType::RowMajorPlanar);
                    continue;
                }

            }

            // Constant is a parameter tensor for a software layer
            //opIt->set<Order>("order", OrderType::RowMajorPlanar);
            opIt->getOutputTensor(0)->setOrder(OrderType::RowMajorPlanar);

        }

    }

}

//NOTE: This should not be done in such hardcoded way.
void addConversionLayersFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    std::cout << "addConversionLayers " << std::endl;

    using namespace mv;

    DataModel dm(model);
    OpModel om(model);

    auto flowIt = dm.flowBegin();

    while (flowIt != dm.flowEnd())
    {
        auto source = flowIt.source();
        auto sink = flowIt.sink();

        //Mandatory check, otherwise the loop could be potentially infinite
        if(source->getOpType() == OpType::Conversion || sink->getOpType() == OpType::Conversion)
        {
            ++flowIt;
            continue;
        }

        //A conversion layer is needed in two cases
        //-p-> stands for a padded tensor
        //1) HW -p-> SW (Target order is RowInterleaved). In this case HW -p-> CONVERSION -> SW
        //2) SW -p-> HW (Target order is planar). In this case SW -> CONVERSION -p-> HW.

        //Reasonable assumption: this pass is executed after the hw marking pass.
        if(!source->hasAttr("NCE1_Compatible") || !sink->hasAttr("NCE1_Compatible"))
        {
            ++flowIt;
            continue;

        }

        // Separate pass for alignment of constant
        if (source->getOpType() == OpType::Constant)
        {
            ++flowIt;
            continue;
        }

        int sourceIsHw = source->get<int>("NCE1_Compatible");
        int sourceIsSw = !sourceIsHw;
        int sinkIsHw = sink->get<int>("NCE1_Compatible");
        int sinkIsSw = !sinkIsHw;
        bool conversionNeeded = false;
        Order targetOrder = OrderType::RowInterleaved;

        //Case 1
        if(sourceIsHw && sinkIsSw)
        {
            targetOrder = OrderType::RowMajorPlanar;
            conversionNeeded = true;
        }

        //Case 2
        if(sourceIsSw && sinkIsHw)
        {
            targetOrder = OrderType::RowInterleaved;
            conversionNeeded = true;
        }

        if(conversionNeeded && !(source->getOpType() == OpType::Input) && !(sink->getOpType() == OpType::Output))
        {
            mv::Data::TensorIterator originalTensor = flowIt->getTensor();
            mv::Data::TensorIterator conversionOutputTensor = om.conversion(originalTensor, targetOrder);

            //If the tensor we are "splitting" through the conversion layer has paddings, they must be handled.
            //Case1 (HW -> SW): Original tensor keeps it's padding, new tensor gets no padding
            //Case2 (SW -> HW): New tensor needs padding, original tensor doesn't need them anymore -> Paddings must be moved
            if(sourceIsSw && sinkIsHw)
            {
                if(originalTensor->hasAttr("NCE1_Paddings"))
                {
                    std::vector<std::size_t> paddings = originalTensor->get<std::vector<std::size_t>>("NCE1_Paddings");
                    conversionOutputTensor->set<std::vector<std::size_t>>("NCE1_Paddings", paddings);
                    originalTensor->erase("NCE1_Paddings");
                }
            }

            unsigned i = 0;
            for(; i < sink->inputSlots(); ++i)
                if(sink->getInputTensor(i) == flowIt->getTensor())
                    break;

            //Necessary for iterator validity despite remotion
            auto flowToEliminate = flowIt;
            ++flowIt;
            om.undefineFlow(flowToEliminate);
            sink->erase(std::string("input") + std::to_string(i));
            om.defineFlow(conversionOutputTensor, sink, i);

            for(; i < sink->outputSlots(); ++i)
                sink->getOutputTensor(i)->setOrder(targetOrder);
        }
        else
        {

            // Align memory order when no conversion is needed
            /// Software ops
            if (sourceIsSw && sinkIsSw)
                flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);
            // Hardware ops
            else if (sourceIsHw && sinkIsHw)
                flowIt->getTensor()->setOrder(OrderType::RowInterleaved);

            ++flowIt;

        }

    }

}
