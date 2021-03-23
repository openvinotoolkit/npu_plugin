#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/algorithms/transitive_reduction.hpp"
#include "include/mcm/algorithms/critical_path.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/scheduling_sort.hpp"

mv::ControlModel::ControlModel(ComputationModel &other) :
ComputationModel(other)
{

}

mv::ControlModel::~ControlModel()
{

}

mv::Control::OpListIterator mv::ControlModel::switchContext(Data::OpListIterator other)
{
    return opsGraph_->get_second_iterator(other);
}

mv::Control::OpListIterator mv::ControlModel::getFirst()
{
   conjoined_graph<Op, DataFlow, ControlFlow>::first_graph::node_list_iterator it = *input_;
   return opsGraph_->get_second_iterator(it);
}

mv::Control::OpListIterator mv::ControlModel::getLast()
{
   conjoined_graph<Op, DataFlow, ControlFlow>::first_graph::node_list_iterator it = *output_;
   return opsGraph_->get_second_iterator(it);
}

mv::Control::OpListIterator mv::ControlModel::opBegin()
{
    return controlGraph_.node_begin();
}

mv::Control::OpListIterator mv::ControlModel::opEnd()
{
    return *controlOpEnd_;
}

mv::Control::FlowListIterator mv::ControlModel::getInput()
{
    return switchContext(*input_).leftmostOutput();
}

mv::Control::FlowListIterator mv::ControlModel::getOutput()
{
    return switchContext(*output_).leftmostInput();
}

mv::Control::FlowListIterator mv::ControlModel::flowBegin()
{
    return controlGraph_.edge_begin();
}

mv::Control::FlowListIterator mv::ControlModel::flowEnd()
{
    return *controlFlowEnd_;
}

void mv::ControlModel::addGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including op to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including op to a group");

    group->include(element);
}

void mv::ControlModel::addGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including control flow to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including control flow to a group");

    group->include(element);
}

void mv::ControlModel::removeGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    group->exclude(element);
}

void mv::ControlModel::removeGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    group->exclude(element);
}

mv::Control::StageIterator mv::ControlModel::addStage()
{
    auto it = stages_->emplace(stages_->size(), std::make_shared<Stage>(*this, stages_->size()));
    return it.first;
}

mv::Control::StageIterator mv::ControlModel::getStage(std::size_t stageIdx)
{
    return stages_->find(stageIdx);
}

void mv::ControlModel::removeStage(Control::StageIterator stage)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed for stage deletion");

    stage->clear();
    stages_->erase(stage->getIdx());

}

void mv::ControlModel::addToStage(Control::StageIterator stage, Control::OpListIterator op)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed during appending an op to a stage");

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed during appending an op to a stage");

    stage->include(op);

}

void mv::ControlModel::addToStage(Control::StageIterator stage, Data::OpListIterator op)
{
    addToStage(stage, switchContext(op));
}

void mv::ControlModel::removeFromStage(Control::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "stage", "invalid", "Invalid op iterator passed during removing an op from a stage");

    if (!op->hasAttr("stage"))
        throw ArgumentError(*this, "op", "invalid", "Attempt of removing an unassigned op from a stage");

    auto stage = getStage(op->get<std::size_t>("stage"));
    stage->exclude(op);

}

std::size_t mv::ControlModel::stageSize() const
{
    return stages_->size();
}

mv::Control::StageIterator mv::ControlModel::stageBegin()
{
    return stages_->begin();
}

mv::Control::StageIterator mv::ControlModel::stageEnd()
{
    return stages_->end();
}

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp)
{

    if (!isValid(sourceOp))
        return flowEnd();

    if (!isValid(sinkOp))
        return flowEnd();

    Control::FlowListIterator flow = controlGraph_.edge_insert(sourceOp, sinkOp, ControlFlow(*this, sourceOp, sinkOp));

    if (flow != *controlFlowEnd_)
    {
        controlFlows_->emplace(flow->getName(), flow);
        log(Logger::MessageType::Debug, "Defined " + flow->toString());
        return flow;
    }
    else
    {
        log(Logger::MessageType::Error, "Unable to define new control flow between " +
            sourceOp->getName() + " and " + sinkOp->getName());
    }

    return flowEnd();

}

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp)
{
   return defineFlow(switchContext(sourceOp), switchContext(sinkOp));
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::topologicalSort()
{
    // Necessary for correct iterator casting
    auto topologicalSortResult = mv::topologicalSort(controlGraph_);
    std::vector<mv::Control::OpListIterator> toReturn(topologicalSortResult.begin(), topologicalSortResult.end());
    return toReturn;
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::schedulingSort()
{
    std::vector<mv::Control::OpListIterator> toSort;
    for(auto opIt = opBegin(); opIt != opEnd(); ++opIt)
        if(opIt->hasAttr("schedulingNumber"))
            toSort.push_back(opIt);

    std::sort(toSort.begin(), toSort.end(), [](mv::Control::OpListIterator a, mv::Control::OpListIterator b){
        unsigned schedulingNumberA = a->get<unsigned>("schedulingNumber");
        unsigned schedulingNumberB = b->get<unsigned>("schedulingNumber");

        return schedulingNumberA < schedulingNumberB;
    });

    return toSort;
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::schedulingSortDPUorUPA()
{
    std::vector<mv::Control::OpListIterator> toSort;
    for(auto opIt = opBegin(); opIt != opEnd(); ++opIt)
        if(opIt->hasAttr("schedulingNumber") && (opIt->getOpType() == "DPUTask" || opIt->getOpType() == "UPATask"))
            toSort.push_back(opIt);

    std::sort(toSort.begin(), toSort.end(), [](mv::Control::OpListIterator a, mv::Control::OpListIterator b){
        unsigned schedulingNumberA = a->get<unsigned>("schedulingNumber");
        unsigned schedulingNumberB = b->get<unsigned>("schedulingNumber");

        return schedulingNumberA < schedulingNumberB;
    });

    return toSort;
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::schedulingSortDMA()
{
    std::vector<mv::Control::OpListIterator> toSort;
    for(auto opIt = opBegin(); opIt != opEnd(); ++opIt)
        if(opIt->hasAttr("schedulingNumber") && (opIt->getOpType() == "DMATask"))
            toSort.push_back(opIt);

    std::sort(toSort.begin(), toSort.end(), [](mv::Control::OpListIterator a, mv::Control::OpListIterator b){
        unsigned schedulingNumberA = a->get<unsigned>("schedulingNumber");
        unsigned schedulingNumberB = b->get<unsigned>("schedulingNumber");

        return schedulingNumberA < schedulingNumberB;
    });

    return toSort;
}

struct OpItComparator
{
    bool operator()(mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator lhs, mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

struct EdgeItComparator
{
    bool operator()(mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator lhs, mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

std::vector<mv::Control::FlowListIterator> mv::ControlModel::criticalPath(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp, const std::string& nodeAttribute, const std::string& edgeAttribute)
{
    std::map<mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator, unsigned, OpItComparator> nodeCosts;
    std::map<mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator, unsigned, EdgeItComparator> edgeCosts;

    if(nodeAttribute != "")
        for(auto opIt = opBegin(); opIt != opEnd(); ++opIt)
            if(opIt->hasAttr(nodeAttribute))
                nodeCosts[opIt] = opIt->get<unsigned>(nodeAttribute);

    if(edgeAttribute != "")
        for(auto edgeIt = flowBegin(); edgeIt != flowEnd(); ++edgeIt)
            if(edgeIt->hasAttr(edgeAttribute))
                edgeCosts[edgeIt] = edgeIt->get<unsigned>(edgeAttribute);

    auto toReturnToBeCasted = mv::critical_path<Op, ControlFlow, OpItComparator, EdgeItComparator>(controlGraph_, sourceOp, sinkOp, nodeCosts, edgeCosts);
    std::vector<mv::Control::FlowListIterator> toReturn(toReturnToBeCasted.begin(), toReturnToBeCasted.end());
    return toReturn;
}

std::vector<mv::Control::FlowListIterator> mv::ControlModel::criticalPath(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp, const std::string& nodeAttribute, const std::string& edgeAttribute)
{
   return criticalPath(switchContext(sourceOp), switchContext(sinkOp), nodeAttribute, edgeAttribute);
}

void mv::ControlModel::transitiveReduction(const std::string& edgeAttribute)
{
    std::set<mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator, EdgeItComparator> toSave;
    if(edgeAttribute != "")
        for(auto edgeIt = flowBegin(); edgeIt != flowEnd(); ++edgeIt)
            if(edgeIt->hasAttr(edgeAttribute))
                toSave.insert(edgeIt);
    mv::transitiveReduction<Op, ControlFlow, EdgeItComparator, OpItComparator>(controlGraph_, toSave);
}

bool mv::ControlModel::isDag()
{
    return mv::isDAG(controlGraph_);
}

mv::mish_params_t mv::ControlModel::getMishParameters(const double maxQuant) {
    const std::map<int32_t, mish_params_t> MISH_PARAMS = {
        {65000, {
            { -128, -104, -77, -52, -33, -9, 6, 49, 127},
            { 4, 5, 4, 4, 4, 0, -1, -1,},
            { 8, 2, 1, -6, -9, 1, -4, 0,},
            1,
        }},
        {66757, {
            { -128, -101, -74, -50, -31, -9, 6, 47, 127},
            { 1, 1, 1, 1, 0, -3, -4, -4,},
            { 64, 35, 5, -55, -65, 8, -32, 0,},
            4,
        }},
        {73281, {
            { -128, -90, -65, -47, -33, -9, 6, 36, 127},
            { 3, 2, 2, 1, 2, -2, -3, -3,},
            { 16, 15, 1, -8, -29, 4, -16, 0,},
            3,
        }},
        {74804, {
            { -128, -90, -65, -47, -33, -9, 6, 36, 127},
            { 3, 2, 2, 1, 2, -2, -3, -3,},
            { 16, 15, 1, -8, -29, 4, -16, 0,},
            3,
        }},
        {74805, {
            { -128, -90, -65, -47, -33, -9, 6, 36, 127},
            { 3, 2, 2, 1, 2, -2, -3, -3,},
            { 16, 15, 1, -8, -29, 4, -16, 0,},
            3,
        }},
        {76484, {
            { -128, -85, -61, -44, -30, -8, 6, 35, 127},
            { 3, 2, 2, 1, 2, -2, -3, -3,},
            { 16, 14, 0, -10, -29, 4, -16, 0,},
            3,
        }},
        {76718, {
            { -128, -85, -61, -44, -30, -8, 6, 35, 127},
            { 3, 2, 2, 1, 2, -2, -3, -3,},
            { 16, 14, 0, -10, -29, 4, -16, 0,},
            3,
        }},
        {81015, {
            { -128, -79, -57, -45, -30, -8, 6, 32, 127},
            { 3, 2, 1, 1, 2, -2, -3, -3,},
            { 16, 12, 13, -1, -29, 4, -16, 0,},
            3,
        }},
        {81484, {
            { -128, -79, -56, -44, -30, -8, 6, 31, 127},
            { 3, 2, 1, 1, 2, -2, -3, -3,},
            { 16, 12, 12, -2, -29, 4, -16, 0,},
            3,
        }},
        {81875, {
            { -128, -78, -56, -44, -30, -8, 6, 31, 127},
            { 3, 2, 1, 1, 1, -2, -3, -3,},
            { 16, 12, 12, -2, -25, 4, -16, 0,},
            3,
        }},
        {82656, {
            { -128, -77, -55, -43, -29, -8, 6, 31, 127},
            { 2, 1, 0, 0, 0, -3, -4, -4,},
            { 32, 23, 23, -5, -51, 8, -32, 0,},
            4,
        }},
        {82812, {
            { -128, -77, -55, -43, -29, -8, 6, 31, 127},
            { 2, 1, 0, 0, 0, -3, -4, -4,},
            { 32, 23, 23, -5, -51, 8, -32, 0,},
            4,
        }},
        {82813, {
            { -128, -77, -55, -43, -29, -8, 6, 31, 127},
            { 2, 1, 0, 0, 0, -3, -4, -4,},
            { 32, 23, 23, -5, -51, 8, -32, 0,},
            4,
        }},
        {86953, {
            { -128, -73, -51, -36, -19, -8, 6, 29, 127},
            { 3, 2, 1, 2, 0, -2, -3, -3,},
            { 16, 11, 10, -23, -17, 4, -16, 0,},
            3,
        }},
        {88437, {
            { -128, -71, -50, -39, -30, -5, 6, 29, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 5, -2, -11, 2, -8, 0,},
            2,
        }},
        {89219, {
            { -128, -70, -49, -38, -29, -5, 6, 28, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 5, -2, -11, 2, -8, 0,},
            2,
        }},
        {90000, {
            { -128, -69, -49, -38, -29, -5, 6, 27, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 5, -2, -11, 2, -8, 0,},
            2,
        }},
        {91484, {
            { -128, -68, -48, -37, -28, -5, 6, 27, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -2, -11, 2, -8, 0,},
            2,
        }},
        {91875, {
            { -128, -68, -47, -36, -27, -5, 6, 27, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {92343, {
            { -128, -67, -47, -36, -27, -5, 6, 26, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {92344, {
            { -128, -67, -47, -36, -27, -5, 6, 26, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {93515, {
            { -128, -66, -46, -35, -26, -5, 6, 26, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {93516, {
            { -128, -66, -46, -35, -26, -5, 6, 26, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {94687, {
            { -128, -65, -45, -35, -26, -5, 6, 25, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {94688, {
            { -128, -65, -45, -35, -26, -5, 6, 25, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {95781, {
            { -128, -64, -45, -34, -25, -5, 6, 25, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 4, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {96171, {
            { -128, -64, -44, -34, -25, -5, 6, 25, 127},
            { 2, 1, 0, 0, 0, -3, -4, -4,},
            { 32, 16, 12, -14, -39, 8, -32, 0,},
            4,
        }},
        {96172, {
            { -128, -64, -44, -34, -25, -5, 6, 25, 127},
            { 2, 1, 0, 0, 0, -3, -4, -4,},
            { 32, 16, 12, -14, -39, 8, -32, 0,},
            4,
        }},
        {96640, {
            { -128, -63, -44, -33, -24, -5, 6, 25, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 16, 12, -15, -39, 8, -32, 0,},
            4,
        }},
        {96641, {
            { -128, -63, -44, -33, -24, -5, 6, 25, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 16, 12, -15, -39, 8, -32, 0,},
            4,
        }},
        {97031, {
            { -128, -63, -44, -33, -24, -5, 6, 25, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 16, 12, -15, -39, 8, -32, 0,},
            4,
        }},
        {98281, {
            { -128, -62, -43, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 11, -16, -39, 8, -32, 0,},
            4,
        }},
        {98437, {
            { -128, -62, -43, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 11, -16, -39, 8, -32, 0,},
            4,
        }},
        {98438, {
            { -128, -62, -43, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 11, -16, -39, 8, -32, 0,},
            4,
        }},
        {98984, {
            { -128, -62, -42, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 10, -16, -39, 8, -32, 0,},
            4,
        }},
        {99140, {
            { -128, -61, -42, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 10, -16, -39, 8, -32, 0,},
            4,
        }},
        {99141, {
            { -128, -61, -42, -32, -23, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 10, -16, -39, 8, -32, 0,},
            4,
        }},
        {99843, {
            { -128, -61, -42, -32, -22, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 10, -16, -39, 8, -32, 0,},
            4,
        }},
        {99844, {
            { -128, -61, -42, -32, -22, -5, 6, 24, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 15, 10, -16, -39, 8, -32, 0,},
            4,
        }},
        {101641, {
            { -128, -60, -41, -31, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 9, -17, -39, 8, -32, 0,},
            4,
        }},
        {101875, {
            { -128, -59, -41, -30, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 9, -18, -39, 8, -32, 0,},
            4,
        }},
        {102578, {
            { -128, -59, -40, -30, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 8, -18, -39, 8, -32, 0,},
            4,
        }},
        {103280, {
            { -128, -58, -40, -30, -20, -5, 6, 22, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 13, 8, -18, -39, 8, -32, 0,},
            4,
        }},
        {107422, {
            { -128, -55, -38, -28, -18, -5, 6, 20, 127},
            { 5, 3, 2, 2, 1, -1, -2, -2,},
            { 4, 3, 2, -5, -7, 2, -8, 0,},
            2,
        }},
        {109453, {
            { -128, -54, -37, -27, -17, -4, 6, 20, 127},
            { 4, 2, 1, 1, 0, -2, -3, -3,},
            { 8, 6, 3, -10, -13, 4, -8, 0,},
            3,
        }},
        {112266, {
            { -128, -52, -35, -26, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 20, 6, -59, 16, -64, -32, 0,},
            5,
        }},
        {112500, {
            { -128, -52, -35, -25, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 20, 6, -59, 16, -64, -32, 0,},
            5,
        }},
        {113047, {
            { -128, -52, -35, -25, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 20, 6, -59, 16, -64, -32, 0,},
            5,
        }},
        {114375, {
            { -128, -51, -34, -25, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 19, 4, -59, 16, -64, -32, 0,},
            5,
        }},
        {116641, {
            { -128, -50, -33, -24, -2, 5, 10, 24, 127},
            { 2, 0, -1, 0, -4, -4, -5, -5,},
            { 32, 18, 2, -59, 16, 80, -32, 0,},
            5,
        }},
        {122655, {
            { -128, -47, -31, -21, -5, 0, 3, 23, 127},
            { 6, 4, 3, 4, 1, 0, -1, -1,},
            { 2, 1, 0, -4, -1, 1, -2, 0,},
            1,
        }},
        {124375, {
            { -128, -46, -30, -21, -5, 0, 3, 22, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 15, -1, -13, -4, 4, -8, 0,},
            3,
        }},
        {124766, {
            { -128, -46, -30, -21, -5, 0, 3, 22, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 15, -1, -13, -4, 4, -8, 0,},
            3,
        }},
        {131719, {
            { -128, -43, -28, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 14, -2, -13, -4, 4, -8, 0,},
            3,
        }},
        {131797, {
            { -128, -43, -28, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 14, -2, -13, -4, 4, -8, 0,},
            3,
        }},
        {132266, {
            { -128, -42, -27, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 13, -2, -13, -4, 4, -8, 0,},
            3,
        }},
        {133281, {
            { -128, -42, -27, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 13, -2, -13, -2, 4, -8, 0,},
            3,
        }},
        {150234, {
            { -128, -36, -22, -13, -5, 0, 3, 17, 127},
            { 4, 1, 1, 0, -1, -2, -3, -3,},
            { 8, 10, -5, -9, -2, 4, -8, 0,},
            3,
        }},
        {153047, {
            { -128, -35, -22, -12, -5, 0, 3, 17, 127},
            { 4, 1, 1, 0, -1, -2, -3, -3,},
            { 8, 10, -5, -9, -2, 4, -8, 0,},
            3,
        }},
        {153828, {
            { -128, -35, -21, -11, -5, 0, 3, 17, 127},
            { 5, 2, 2, 0, 0, -1, -2, -2,},
            { 4, 5, -2, 0, -1, 2, -4, 0,},
            2,
        }},
        {160938, {
            { -128, -33, -20, -11, -5, 0, 3, 16, 127},
            { 5, 2, 2, 1, 0, -1, -2, -2,},
            { 4, 5, -3, -2, -1, 2, -4, 0,},
            2,
        }},
        {161719, {
            { -128, -32, -20, -11, -5, 0, 3, 15, 127},
            { 5, 2, 2, 1, 0, -1, -2, -2,},
            { 4, 4, -3, -2, -1, 2, -4, 0,},
            2,
        }},
        {161875, {
            { -128, -32, -20, -11, -5, 0, 3, 15, 127},
            { 5, 2, 2, 1, 0, -1, -2, -2,},
            { 4, 4, -3, -2, -1, 2, -4, 0,},
            2,
        }},
        {164375, {
            { -128, -32, -19, -11, -5, 0, 3, 15, 127},
            { 5, 2, 2, 1, 0, -1, -2, -2,},
            { 4, 4, -3, -2, -1, 2, -4, 0,},
            2,
        }},
        {169531, {
            { -128, -30, -18, -11, -5, 0, 3, 14, 127},
            { 5, 2, 1, 1, 0, -1, -2, -2,},
            { 4, 4, 1, -2, -1, 2, -4, 0,},
            2,
        }},
        {178438, {
            { -128, -28, -17, -11, -5, 0, 3, 13, 127},
            { 5, 2, 1, 1, 0, -1, -2, -2,},
            { 4, 3, 1, -2, -1, 2, -4, 0,},
            2,
        }},
        {189061, {
            { -128, -26, -15, -11, -5, 0, 3, 12, 127},
            { 5, 2, 0, 1, 0, -1, -2, -2,},
            { 4, 3, 7, -2, -1, 2, -4, 0,},
            2,
        }},
        {192188, {
            { -128, -26, -14, -11, -5, 0, 3, 12, 127},
            { 5, 2, 0, 1, 0, -1, -2, -2,},
            { 4, 3, 6, -2, -1, 2, -4, 0,},
            2,
        }},
        {192656, {
            { -128, -26, -14, -11, -5, 0, 3, 12, 127},
            { 5, 2, 0, 1, 0, -1, -2, -2,},
            { 4, 3, 6, -2, -1, 2, -4, 0,},
            2,
        }},
        {193594, {
            { -128, -25, -14, -11, -5, 0, 3, 12, 127},
            { 5, 2, 0, 1, 0, -1, -2, -2,},
            { 4, 3, 6, -2, -1, 2, -4, 0,},
            2,
        }},
        {198281, {
            { -128, -25, -13, -11, -5, 0, 3, 12, 127},
            { 5, 2, -1, 1, 0, -1, -2, -2,},
            { 4, 3, 18, -2, -1, 2, -4, 0,},
            2,
        }},
        {198282, {
            { -128, -25, -13, -11, -5, 0, 3, 12, 127},
            { 5, 2, -1, 1, 0, -1, -2, -2,},
            { 4, 3, 18, -2, -1, 2, -4, 0,},
            2,
        }},
        {215156, {
            {-128, -22, -11, -2, 0, 2, 10, 124, 127},
            {4, 1, 0, -2, -3, -3, -3, -3},
            {8, 3, -5, 0, 0, -8, 0, 0},
            3,
        }},
        {237500, {
            {-128, -19, -9, -2, 0, 2, 9, 124, 127},
            {6, 3, 2, 0, -1, -1, -1, -1},
            {2, 1, -1, 0, 0, -2, 0, 0},
            1,
        }},
        {238438, {
            {-128, -19, -9, -2, 0, 2, 9, 124, 127},
            {6, 3, 2, 0, -1, -1, -1, -1},
            {2, 1, -1, 0, 0, -2, 0, 0},
            1,
        }},
        {254844, {
            {-128, -17, -7, -2, 0, 2, 8, 124, 127},
            {7, 5, 2, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {269531, {
            {-128, -16, -7, -2, 0, 2, 7, 124, 127},
            {7, 4, 3, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {307500, {
            {-128, -13, -7, -2, 0, 2, 6, 124, 127},
            {7, 4, 3, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {355312, {
            {-128, -11, -7, 0, 3, 4, 6, 124, 127},
            {7, 4, 3, 0, -5, 0, 0, 0},
            {1, 0, 0, 0, -94, 0, 0, 0},
            0,
        }},
        {355313, {
            {-128, -11, -7, 0, 3, 4, 6, 124, 127},
            {7, 4, 3, 0, -5, 0, 0, 0},
            {1, 0, 0, 0, -94, 0, 0, 0},
            0,
        }},
        {388125, {
            {-128, -9, -7, 0, 3, 4, 6, 124, 127},
            {7, 4, 3, 0, -5, 0, 0, 0},
            {1, 0, 0, 0, -93, 0, 0, 0},
            0,
        }},

        {112253, {
            { -128, -52, -35, -26, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 20, 6, -59, 16, -64, -32, 0,},
            5,
        }},
        {113965, {
            { -128, -51, -34, -25, -2, 6, 12, 25, 127},
            { 2, 0, -1, 0, -4, -5, -5, -5,},
            { 32, 19, 4, -59, 16, -64, -32, 0,},
            5,
        }},
        {91865, {
            { -128, -68, -47, -36, -27, -5, 6, 27, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {88441, {
            { -128, -71, -50, -39, -30, -5, 6, 29, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 5, -2, -11, 2, -8, 0,},
            2,
        }},
        {133264, {
            { -128, -42, -27, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 13, -2, -13, -2, 4, -8, 0,},
            3,
        }},
        {261082, {
            {-128, -17, -7, -2, 0, 2, 8, 124, 127},
            {7, 5, 2, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {228631, {
            {-128, -19, -9, -2, 0, 2, 9, 124, 127},
            {6, 3, 2, 0, -1, -1, -1, -1},
            {2, 1, -1, 0, 0, -2, 0, 0},
            1,
        }},
        {216725, {
            {-128, -22, -11, -2, 0, 2, 10, 124, 127},
            {4, 1, 0, -2, -3, -3, -3, -3},
            {8, 3, -5, 0, 0, -8, 0, 0},
            3,
        }},
        {101787, {
            { -128, -60, -41, -31, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 9, -17, -39, 8, -32, 0,},
            4,
        }},
        {97935, {
            { -128, -63, -44, -33, -24, -5, 6, 25, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 16, 12, -15, -39, 8, -32, 0,},
            4,
        }},
        {94666, {
            { -128, -65, -45, -35, -26, -5, 6, 25, 127},
            { 4, 3, 2, 2, 3, -1, -2, -2,},
            { 8, 5, 4, -3, -11, 2, -8, 0,},
            2,
        }},
        {161863, {
            { -128, -32, -20, -11, -5, 0, 3, 15, 127},
            { 5, 2, 2, 1, 0, -1, -2, -2,},
            { 4, 4, -3, -2, -1, 2, -4, 0,},
            2,
        }},
        {285828, {
            {-128, -16, -7, -2, 0, 2, 7, 124, 127},
            {7, 4, 3, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {206375, {
            { -128, -25, -13, -11, -5, 0, 3, 12, 127},
            { 5, 2, -1, 1, 0, -1, -2, -2,},
            { 4, 3, 18, -2, -1, 2, -4, 0,},
            2,
        }},
        {205441, {
            { -128, -25, -13, -11, -5, 0, 3, 12, 127},
            { 5, 2, -1, 1, 0, -1, -2, -2,},
            { 4, 3, 18, -2, -1, 2, -4, 0,},
            2,
        }},
        {170423, {
            { -128, -30, -18, -11, -5, 0, 3, 14, 127},
            { 5, 2, 1, 1, 0, -1, -2, -2,},
            { 4, 4, 1, -2, -1, 2, -4, 0,},
            2,
        }},
        {137817, {
            { -128, -42, -27, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 13, -2, -13, -2, 4, -8, 0,},
            3,
        }},
        {107429, {
            { -128, -55, -38, -28, -18, -5, 6, 20, 127},
            { 5, 3, 2, 2, 1, -1, -2, -2,},
            { 4, 3, 2, -5, -7, 2, -8, 0,},
            2,
        }},
        {130938, {
            { -128, -43, -28, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 14, -2, -13, -4, 4, -8, 0,},
            3,
        }},
        {269564, {
            {-128, -16, -7, -2, 0, 2, 7, 124, 127},
            {7, 4, 3, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {102020, {
            { -128, -59, -41, -30, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 9, -18, -39, 8, -32, 0,},
            4,
        }},
        {254856, {
            {-128, -17, -7, -2, 0, 2, 8, 124, 127},
            {7, 5, 2, 1, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, -1, 0, 0},
            0,
        }},
        {232989, {
            {-128, -19, -9, -2, 0, 2, 9, 124, 127},
            {6, 3, 2, 0, -1, -1, -1, -1},
            {2, 1, -1, 0, 0, -2, 0, 0},
            1,
        }},
        {103265, {
            { -128, -59, -41, -30, -21, -5, 6, 23, 127},
            { 3, 1, 0, 0, 0, -3, -4, -4,},
            { 16, 14, 9, -18, -39, 8, -32, 0,},
            4,
        }},
        {192601, {
            { -128, -26, -14, -11, -5, 0, 3, 12, 127},
            { 5, 2, 0, 1, 0, -1, -2, -2,},
            { 4, 3, 6, -2, -1, 2, -4, 0,},
            2,
        }},
        {207231, {
            { -128, -25, -13, -11, -5, 0, 3, 12, 127},
            { 5, 2, -1, 1, 0, -1, -2, -2,},
            { 4, 3, 18, -2, -1, 2, -4, 0,},
            2,
        }},
        {123109, {
            { -128, -47, -31, -21, -5, 0, 3, 23, 127},
            { 6, 4, 3, 4, 1, 0, -1, -1,},
            { 2, 1, 0, -4, -1, 1, -2, 0,},
            1,
        }},
        {140000, {
            { -128, -42, -27, -18, -5, 0, 3, 20, 127},
            { 4, 1, 1, 1, -1, -2, -3, -3,},
            { 8, 13, -2, -13, -2, 4, -8, 0,},
            3,
        }},
    };

    int32_t max_quant = std::round(maxQuant * 10000.f);
    const auto mish_params = MISH_PARAMS.find(max_quant);
    if (mish_params == MISH_PARAMS.end()) {
        throw std::runtime_error("getMishParameters: Couldn't find Mish parameters for " + std::to_string(maxQuant));
    }
    return mish_params->second;
}

mv::Control::OpListIterator mv::ControlModel::cycleResponsible()
{
    return  mv::getNodeInCycle(controlGraph_).second;
}

void mv::ControlModel::undefineFlow(Control::FlowListIterator flow)
{

    if (!ComputationModel::isValid(flow))
        throw ArgumentError(*this, "flow", "invalid", "An invalid flow iterator passed for flow deletion");

    controlFlows_->erase(flow->getName());
    controlGraph_.edge_erase(flow);

}

std::string mv::ControlModel::getLogID() const
{
    return "ControlModel:" + name_;
}

mv::Control::FlowListIterator mv::ControlModel::checkControlFlow(mv::Control::OpListIterator source, mv::Control::OpListIterator sink)
{
    mv::Control::FlowListIterator toReturn = flowEnd();

    for(auto outFlow = source.leftmostOutput(); outFlow != flowEnd(); ++outFlow)
    {
        if(outFlow.sink() == sink)
        {
            toReturn = outFlow;
            break;
        }
    }

    return toReturn;
}

bool mv::ControlModel::pathExists(Control::OpListIterator source, Control::OpListIterator target)
{
    return mv::pathExists(controlGraph_, source, target);
}


mv::Control::FlowListIterator mv::ControlModel::checkControlFlow(mv::Data::OpListIterator source, mv::Data::OpListIterator sink)
{
    return checkControlFlow(switchContext(source), switchContext(sink));
}


bool mv::ControlModel::isFlowAllowed(mv::Control::OpListIterator source, mv::Control::OpListIterator sink)
{
    // Extra check to verify we are not adding a self-edge
    if(source == sink)
        return false;

    // Extra check to enforce constraint: neither to source nor the target of a control flow cannot be non executable
    if((!source->hasTypeTrait("executable")) || (!sink->hasTypeTrait("executable")))
        return false;

    return true;
}

bool mv::ControlModel::isFlowAllowed(mv::Data::OpListIterator source, mv::Data::OpListIterator sink)
{
    return isFlowAllowed(switchContext(source), switchContext(sink));
}

bool mv::ControlModel::isFlowAllowedAndNonExisting(mv::Control::OpListIterator source, mv::Control::OpListIterator sink)
{
    return isFlowAllowed(source,sink) && (checkControlFlow(source, sink) == flowEnd());
}

bool mv::ControlModel::isFlowAllowedAndNonExisting(mv::Data::OpListIterator source, mv::Data::OpListIterator sink)
{
    return isFlowAllowedAndNonExisting(switchContext(source), switchContext(sink));
}

std::size_t mv::ControlModel::controlFlowsCount() const
{
    return controlGraph_.edge_size();
}
