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
        //1) HW -> SW (Target order is columnMajor)
        //2) SW -> HW (Target order is planar)

        //Reasonable assumption: this pass is executed after the hw marking pass.
        int sourceIsHw = source->getAttr("NCE1_Compatible").getContent<int>();
        int sinkIsHw = sink->getAttr("NCE1_Compatible").getContent<int>();
        bool conversionNeeded = false;
        mv::Order targetOrder = mv::Order::Unknown;

        //Case 1
        if(sourceIsHw && !sinkIsHw)
        {
            targetOrder = mv::Order::RowMajor;
            conversionNeeded = true;
        }

        //Case 2
        if(!sourceIsHw && sinkIsHw)
        {
            targetOrder = mv::Order::Planar;
            conversionNeeded = true;
        }

        if(conversionNeeded && source->getOpType() == mv::OpType::Constant)
        {
            //No need for a conversion layer in this case, just reorder the tensor in place
            flowIt->getTensor()->reorder(targetOrder);
            ++flowIt;
            continue;
        }

        if(conversionNeeded)
        {
            mv::Data::TensorIterator conversionOutputTensor = om.conversion(flowIt->getTensor(), targetOrder);

            unsigned i = 0;
            for(; i < sink->inputSlots(); ++i)
                if(sink->getInputTensor(i) == flowIt->getTensor())
                    break;
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
