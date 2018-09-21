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
        //1) HW -> SW (Target order is columnMajor)
        //2) SW -> HW (Target order is planar)

        //Reasonable assumption: this pass is executed after the hw marking pass.
        int sourceIsHw = source->get<int>("NCE1_Compatible");
        int sinkIsHw = sink->get<int>("NCE1_Compatible");
        bool conversionNeeded = false;
        Order targetOrder = OrderType::ColumnMajor;

        //Case 1
        if(sourceIsHw && !sinkIsHw)
        {
            targetOrder = OrderType::RowMajor;
            conversionNeeded = true;
        }

        //Case 2
        if(!sourceIsHw && sinkIsHw)
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
            Data::TensorIterator conversionOutputTensor = om.conversion(flowIt->getTensor(), targetOrder);

            unsigned i = 0;
            for(; i < sink->inputSlots(); ++i)
                if(sink->getInputTensor(i) == flowIt->getTensor())
                    break;
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
                flowIt->getTensor()->setOrder(OrderType::RowMajor);
            // Hardware ops
            if (sourceIsHw && sinkIsHw)
                flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);

            ++flowIt;
        
        }
    }

}
