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


void compatibilityResolution(mv::Data::OpListIterator parentIt, mv::OpModel *om);

void compatibilityResolution(mv::Data::OpListIterator parentIt, mv::OpModel *om){

    if(parentIt->getOpType() == mv::OpType::Output) {
        return;
    }



    parentIt->set<unsigned>("traversed_CR", 1);

    auto childIt = parentIt.leftmostChild();


    // Will the DNA test reveal the child's true father? Find out after the break!
    auto paternityTest = childIt.leftmostParent();
    bool all_parents_resolved = true;
    while(1){
        if(paternityTest->getOpType() != mv::OpType::Constant)
            if(!paternityTest->hasAttr("traversed_CR") || paternityTest->get<unsigned>("traversed_CR") != 1){
                // std::cout << !paternityTest->hasAttr("traversed_CR") <<":"<< paternityTest->get<unsigned>("traversed_CR") << std::endl;
                all_parents_resolved = false;
            }
        if(paternityTest == childIt.rightmostParent()) break;
        ++paternityTest;
    }
    if(!all_parents_resolved){
        // Oh! Bad luck. Looks like these parents need to make up before letting their small child
        // go out into the big bad world of recursion.
        return;
    }


    while(1){
        std::cout << "Source: " << parentIt->getName() << " Sink: " << childIt->getName() << std::endl;

        auto source = parentIt;
        auto sink = childIt;

        if(source->getOpType() == mv::OpType::Conversion || sink->getOpType() == mv::OpType::Conversion)
        {
            compatibilityResolution(childIt, om);
            if(childIt == parentIt.rightmostChild()) break;
            ++childIt;
            continue;
        }

        if(!source->hasAttr("NCE1_Compatible") || !sink->hasAttr("NCE1_Compatible"))
        {

            compatibilityResolution(childIt, om);
            if(childIt == parentIt.rightmostChild()) break;
            ++childIt;
            continue;
        }
        if (source->getOpType() == mv::OpType::Constant)
        {

            compatibilityResolution(childIt, om);
            if(childIt == parentIt.rightmostChild()) break;
            ++childIt;
            continue;
        }

        int sourceIsHw = source->get<int>("NCE1_Compatible");
        int sourceIsSw = !sourceIsHw;
        int sinkIsHw = sink->get<int>("NCE1_Compatible");
        int sinkIsSw = !sinkIsHw;
        bool conversionNeeded = false;
        mv::Order targetOrder = mv::OrderType::RowInterleaved;

        //Case 1
        if(sourceIsHw && sinkIsSw)
        {
            targetOrder = mv::OrderType::RowMajorPlanar;
            conversionNeeded = true;
        }

        //Case 2
        if(sourceIsSw && sinkIsHw)
        {
            targetOrder = mv::OrderType::RowInterleaved;
            conversionNeeded = true;
        }

        //Concat as sink case
        if(sink->getOpType() == mv::OpType::Concat){
            if (sourceIsSw){
                // flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);
                childIt->getInputTensor(0)->setOrder(mv::OrderType::RowMajorPlanar);
                childIt->getOutputTensor(0)->setOrder(mv::OrderType::RowMajorPlanar);
            }
            // Hardware ops
            else if (sourceIsHw){
                // flowIt->getTensor()->setOrder(OrderType::RowInterleaved);
                childIt->getInputTensor(0)->setOrder(mv::OrderType::RowInterleaved);
                childIt->getOutputTensor(0)->setOrder(mv::OrderType::RowInterleaved);
                sink->set<int>("NCE1_Compatible", 1);
            }

            conversionNeeded = false;
            // ++flowIt;

            compatibilityResolution(childIt, om);
            if(childIt == parentIt.rightmostChild()) break;
            ++childIt;
            continue;
        }


        if(conversionNeeded && !(source->getOpType() == mv::OpType::Input) && !(sink->getOpType() == mv::OpType::Output))
        {
            // mv::Data::TensorIterator originalTensor = flowIt->getTensor();
            mv::Data::TensorIterator originalTensor = childIt->getInputTensor(0);

            mv::Data::TensorIterator conversionOutputTensor = om->conversion(originalTensor, targetOrder);

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
                // if(sink->getInputTensor(i) == flowIt->getTensor())
                if(sink->getInputTensor(i) == childIt->getInputTensor(0))
                    break;

            //Necessary for iterator validity despite remotion

            auto a = parentIt.leftmostOutput();
            om->defineFlow(conversionOutputTensor, sink, i);
            om->undefineFlow(a);
            sink->erase(std::string("input") + std::to_string(i));
            std::cout << "Conversion Placed." << std::endl;

            for(; i < sink->outputSlots(); ++i)
                sink->getOutputTensor(i)->setOrder(targetOrder);

            // Recurse DFS
            // compatibilityResolution(childIt, om);    // This child has a new parent now
            if (childIt == parentIt.rightmostChild()) break;
            ++childIt;


        }
        else
        {

            // Align memory order when no conversion is needed
            /// Software ops
            if (sourceIsSw && sinkIsSw)
                // flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);
                parentIt->getOutputTensor(0)->setOrder(mv::OrderType::RowMajorPlanar);
            // Hardware ops
            else if (sourceIsHw && sinkIsHw)
                // flowIt->getTensor()->setOrder(OrderType::RowInterleaved);
                parentIt->getOutputTensor(0)->setOrder(mv::OrderType::RowInterleaved);

            // ++flowIt;
            std::cout << "Values aligned." << std::endl;

            // Recurse DFS
            compatibilityResolution(childIt, om);
            std::cout << childIt->getName() << std::endl;
            if (childIt == parentIt.rightmostChild()) break;
            std::cout << childIt->getName() << std::endl;
            ++childIt;

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

    auto opIt = om.opBegin();


    compatibilityResolution(opIt, &om);
    std::cout << "Added. " << std::endl;


}
