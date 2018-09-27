#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void addConversionLayers(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AddConversionLayers)
        .setFunc(addConversionLayers)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass adds conversion layers where needed."
        );

    }

}

//NOTE: This should not be done in such hardcoded way.
void addConversionLayers(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
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
        //1) HW -p-> SW (Target order is columnMajor). In this case HW -p-> CONVERSION -> SW
        //2) SW -p-> HW (Target order is planar). In this case SW -> CONVERSION -p-> HW.

        //Reasonable assumption: this pass is executed after the hw marking pass.
        if(!source->hasAttr("NCE1_Compatible") || !sink->hasAttr("NCE1_Compatible"))
            continue;
        int sourceIsHw = source->get<int>("NCE1_Compatible");
        int sourceIsSw = !sourceIsHw;
        int sinkIsHw = sink->get<int>("NCE1_Compatible");
        int sinkIsSw = !sinkIsHw;
        bool conversionNeeded = false;
        Order targetOrder = OrderType::ColumnMajor;

        //Case 1
        if(sourceIsHw && sinkIsSw)
        {
            targetOrder = OrderType::RowMajor;
            conversionNeeded = true;
        }

        //Case 2
        if(sourceIsSw && sinkIsHw)
        {
            targetOrder = OrderType::RowMajorPlanar;
            conversionNeeded = true;
        }

        if(conversionNeeded && source->getOpType() == OpType::Constant)
        {
            //No need for a conversion layer in this case, just reorder the tensor in place
            flowIt->getTensor()->setOrder(targetOrder);
            ++flowIt;
            continue;
        }

        if(conversionNeeded)
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
        }
        else
        {

            // Align memory order when no conversion is needed
            /// Software ops
            if (!sourceIsHw && !sinkIsHw)
            {
                if (source->getOpType() == OpType::Constant)
                    flowIt->getTensor()->setOrder(OrderType::ColumnMajorPlanar);
                else
                    flowIt->getTensor()->setOrder(OrderType::RowMajor);
            }
            // Hardware ops
            if (sourceIsHw && sinkIsHw)
                flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);

            ++flowIt;

        }
    }

}
