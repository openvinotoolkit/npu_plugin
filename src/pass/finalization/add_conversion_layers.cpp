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
void addConversionLayers(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::DataModel dm(model);
    mv::OpModel om(model);

    auto flowIt = dm.flowBegin();

    while (flowIt != dm.flowEnd())
    {
        auto source = flowIt.source();
        auto sink = flowIt.sink();

        //Mandatory check, otherwise the loop could be potentially infinite
        if(source->getOpType() == mv::OpType::Conversion || sink->getOpType() == mv::OpType::Conversion)
        {
            ++flowIt;
            continue;
        }

        //A conversion layer is needed in two cases
        //-p-> stands for a padded tensor
        //1) HW -p-> SW (Target order is columnMajor). In this case HW -p-> CONVERSION -> SW
        //2) SW -p-> HW (Target order is planar). In this case SW -> CONVERSION -p-> HW.

        //Reasonable assumption: this pass is executed after the hw marking pass.
        int sourceIsHw = source->getAttr("NCE1_Compatible").getContent<int>();
        int sourceIsSw = !sourceIsHw;
        int sinkIsHw = sink->getAttr("NCE1_Compatible").getContent<int>();
        int sinkIsSw = !sinkIsHw;
        bool conversionNeeded = false;
        mv::Order targetOrder = mv::Order::Unknown;

        //Case 1
        if(sourceIsHw && sinkIsSw)
        {
            targetOrder = mv::Order::RowMajor;
            conversionNeeded = true;
        }

        //Case 2
        if(sourceIsSw && sinkIsHw)
        {
            targetOrder = mv::Order::RowMajorPlanar;
            conversionNeeded = true;
        }

        if(conversionNeeded && source->getOpType() == mv::OpType::Constant)
        {
            //No need for a conversion layer in this case, just reorder the tensor in place
            flowIt->getTensor()->reorder(targetOrder);
            source->removeAttr("order");
            om.addAttr(source, "order", mv::Attribute(mv::AttrType::OrderType, targetOrder));
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
                    mv::dynamic_vector<size_t> paddings = originalTensor->getAttr("NCE1_Paddings").getContent<mv::dynamic_vector<size_t>>();
                    dm.addAttr(conversionOutputTensor, "NCE1_Paddings", mv::Attribute(mv::AttrType::UnsignedVecType, paddings));
                    originalTensor->removeAttr("NCE1_Paddings");
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
            sink->removeAttr(std::string("input")+std::to_string(i));
            om.defineFlow(conversionOutputTensor, sink, i);
        }
        else
        {
            ++flowIt;
        }
    }

}
