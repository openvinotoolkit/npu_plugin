#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void addConversionLayersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void alignConstOrderFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void compatibilityResolution(mv::Data::OpListIterator parentIt, mv::OpModel &om);

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

void alignConstOrderFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    DataModel dm(model);
    OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Constant")
        {

            if (opIt.childrenSize() > 1)
                throw OpError(*opIt, "Constant has more than 1 children, currently unsupported");

            // Constant is a parameter tensor for a hardware layer
            if (opIt.leftmostChild()->hasAttr("NCE1_Compatible"))
            {
                if (opIt.leftmostChild()->get<int>("NCE1_Compatible") == 1)
                {
                    //opIt->set<Order>("order", OrderType::RowMajorPlanar);
                    opIt.leftmostOutput()->getTensor()->setOrder(Order(Order::getRowMajorID(opIt.leftmostOutput()->getTensor()->getShape().ndims())));
                    pass.log(Logger::MessageType::Info, "Changed data order of the NCE executed op " + opIt->getName() + " to RowMajorPlanar");
                    continue;
                }

            }

            // Constant is a parameter tensor for a software layer
            //opIt->set<Order>("order", OrderType::RowMajorPlanar);
            //opIt->set<Order>("Order", Order("HWC"));
            if (opIt.leftmostChild()->getOpType() == OpType::FullyConnected)
                opIt->getOutputTensor(0)->setOrder(Order::getRowMajorID(opIt->getOutputTensor(0)->getShape().ndims()));
            else
                opIt->getOutputTensor(0)->setOrder(Order::getRowMajorPlanarID(opIt->getOutputTensor(0)->getShape().ndims()));
            pass.log(Logger::MessageType::Info, "Changed data order of the software executed op " + opIt->getName() + " to RowMajorPlanar");

        }

    }

}




void compatibilityResolution(mv::Data::OpListIterator parentIt, mv::OpModel& om)
{

    if(parentIt->getOpType() == "Output")
        return;

    parentIt->set<unsigned>("traversed_CR", 1);

    auto childIt = parentIt.leftmostChild();

    // Will the DNA test reveal the child's true father? Find out after the break!
    auto paternityTest = childIt.leftmostParent();
    bool all_parents_resolved = true;
    while(1)
    {
        if(paternityTest->getOpType() != "Constant")
            if(!paternityTest->hasAttr("traversed_CR") || paternityTest->get<unsigned>("traversed_CR") != 1)
            {
                // std::cout << !paternityTest->hasAttr("traversed_CR") <<":"<< paternityTest->get<unsigned>("traversed_CR") << std::endl;
                all_parents_resolved = false;
            }
        if(paternityTest == childIt.rightmostParent()) break;
        ++paternityTest;
    }
    if(!all_parents_resolved)
    {
        // Oh! Bad luck. Looks like these parents need to make up before letting their small child
        // go out into the big bad world of recursion.
        return;
    }


    while(1)
    {

        std::cout << "Source: " << parentIt->getName() << " Sink: " << childIt->getName() << std::endl;

        auto source = parentIt;
        auto sink = childIt;

        if(source->getOpType() == "Conversion" || sink->getOpType() == "Conversion")
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
        if (source->getOpType() == "Constant")
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
        mv::Order targetOrder = mv::Order("HCW");

        //Case 1
        if(sourceIsHw && sinkIsSw)
        {
            targetOrder = mv::Order("HWC");
            conversionNeeded = true;
        }

        //Case 2
        if(sourceIsSw && sinkIsHw)
        {
            targetOrder = mv::Order("HCW");
            conversionNeeded = true;
        }

        //Concat as sink case
        if(sink->getOpType() == "Concat")
        {
            if (sourceIsSw)
            {
                // flowIt->getTensor()->setOrder(OrderType::RowMajorPlanar);
                childIt->getInputTensor(0)->setOrder(mv::Order(mv::Order::getRowMajorPlanarID(childIt->getInputTensor(0)->getShape().ndims())));
                childIt->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getRowMajorPlanarID(childIt->getOutputTensor(0)->getShape().ndims())));
            }
            // Hardware ops
            else if (sourceIsHw)
            {
                // flowIt->getTensor()->setOrder(mv::OrderType::RowInterleaved);
                for(unsigned i = 0; i < childIt->inputSlots(); i++)

                    childIt->getInputTensor(i)->setOrder(mv::Order("HCW"));

                childIt->getOutputTensor(0)->setOrder(mv::Order("HCW"));
                sink->set<int>("NCE1_Compatible", 1);
            }

            conversionNeeded = false;
            // ++flowIt;

            compatibilityResolution(childIt, om);
            if(childIt == parentIt.rightmostChild()) break;
            ++childIt;
            continue;
        }


        if(conversionNeeded && !(source->getOpType() == "Input") && !(sink->getOpType() == "Output"))
        {
            // mv::Data::TensorIterator originalTensor = flowIt->getTensor();
            mv::Data::TensorIterator originalTensor = childIt->getInputTensor(0);

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
                // if(sink->getInputTensor(i) == flowIt->getTensor())
                if(sink->getInputTensor(i) == childIt->getInputTensor(0))
                    break;

            //Necessary for iterator validity despite remotion

            auto a = parentIt.leftmostOutput();
            om.defineFlow(conversionOutputTensor, sink, i);
            om.undefineFlow(a);
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
                // flowIt->getTensor()->setOrder(mv::Order("HWC"));
                parentIt->getOutputTensor(0)->setOrder(mv::Order("HWC"));
            // Hardware ops
            else if (sourceIsHw && sinkIsHw)
                // flowIt->getTensor()->setOrder(mv::OrderType::RowInterleaved);
                parentIt->getOutputTensor(0)->setOrder(mv::Order("HWC"));

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
void addConversionLayersFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    std::cout << "addConversionLayers " << std::endl;

    using namespace mv;

    DataModel dm(model);
    OpModel om(model);

    auto opIt = om.opBegin();

    compatibilityResolution(opIt, om);
    std::cout << "Added. " << std::endl;


}
