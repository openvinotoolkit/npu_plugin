#include "include/mcm/pass/serialization/generate_json.hpp"
#include "include/mcm/base/json/json.hpp"

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateJson)
        .setFunc(__generate_json_detail_::generateJsonFcn)
        .setGenre({PassGenre::Validation, PassGenre::Serialization})
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates the json representation of computation model"
        );

    }

}

void mv::pass::__generate_json_detail_::generateJsonFcn(ComputationModel& model, TargetDescriptor&, json::Object& compDesc)
{
    std::ofstream ostream;
    ostream.open(compDesc["GenerateJson"]["output"].get<std::string>(), std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError("output", compDesc["GenerateJson"]["output"].get<std::string>(), "Unable to open output file");

    mv::json::Object computationModel; //The final object to stringify to ostream
    mv::json::Object graph;
    mv::json::Array nodes;
    mv::json::Array edges;
    mv::json::Array tensors;
    mv::json::Array groups;
    //mv::json::Array stages;
    mv::json::Array sourceOps;
    mv::json::Object opsCounters;

    //Deploying graph object
    OpModel opModel(model);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {
        nodes.append(mv::Jsonable::toJsonValue(*opIt));
        opsCounters[mv::Printable::toString(opIt->getOpType())] = mv::Jsonable::toJsonValue(opModel.opsCount(opIt->getOpType()));
    }

    //Data flows
    DataModel dataModel(model);
    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
        for (auto dataIt = opIt.leftmostOutput(); dataIt != dataModel.flowEnd(); ++dataIt)
            edges.append(mv::Jsonable::toJsonValue(*dataIt));

    //Control flows
    ControlModel controlModel(model);
    for (auto opIt = controlModel.getFirst(); opIt != controlModel.opEnd(); ++opIt)
        for (auto controlIt = opIt.leftmostOutput(); controlIt != controlModel.flowEnd(); ++controlIt)
            edges.append(mv::Jsonable::toJsonValue(*controlIt));

    //Deploying tensors
    for (auto tensorIt = opModel.tensorBegin(); tensorIt != opModel.tensorEnd(); ++tensorIt)
    {
        tensors.append(mv::Jsonable::toJsonValue(*tensorIt));
        sourceOps.append(mv::Jsonable::toJsonValue(opModel.getSourceOp(tensorIt)->getName()));
    }

    //Deploying groups
    for (auto groupIt = model.groupBegin(); groupIt != model.groupEnd(); ++groupIt)
        groups.append(mv::Jsonable::toJsonValue(*groupIt));

    //Deploying stages
    //for (auto stagesIt = controlModel.stageBegin(); stagesIt != controlModel.stageEnd(); ++stagesIt)
        //stages.append(mv::Jsonable::toJsonValue(*stagesIt));

    //TODO: Add jsonization for Memory Allocators (currently there is no iterator);

    graph["nodes"] = mv::json::Value(nodes);
    graph["edges"] = mv::json::Value(edges);
    computationModel["graph"] = graph;
    computationModel["tensors"] = tensors;
    computationModel["groups"] = groups;
    //["stages"] = stages;
    computationModel["source_ops"] = sourceOps;
    computationModel["default_control_flow"] = mv::Jsonable::toJsonValue(model.getDefaultControlFlow());
    computationModel["operations_counters"] = opsCounters;
    ostream << computationModel.stringifyPretty();
}
