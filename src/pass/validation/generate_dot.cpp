#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

void generateDotFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateDot)
        .setFunc(generateDotFcn)
        .setGenre({PassGenre::Validation, PassGenre::Serialization})
        .defineArg(json::JSONType::String, "output")
        .defineArg(json::JSONType::String, "scope")
        .defineArg(json::JSONType::String, "content")
        .defineArg(json::JSONType::Bool, "html")
        .setDescription(
            "Generates the DOT representation of computation model"
        );

    }

}

void generateDotFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    using namespace mv;

    if (compDesc["GenerateDot"]["output"].get<std::string>().empty())
        throw ArgumentError("output", "", "Unspecified output name for generate dot pass");

    std::string outputScope = compDesc["GenerateDot"]["scope"].get<std::string>();
    if (outputScope != "OpModel" && outputScope != "ExecOpModel" && outputScope != "ControlModel" &&
        outputScope != "OpControlModel" && outputScope != "ExecOpControlModel" && outputScope != "DataModel")
        throw ArgumentError("scope", outputScope, "Invalid model scope");

    std::string contentLevel = compDesc["GenerateDot"]["content"].get<std::string>();
    if (contentLevel != "full" && outputScope != "name")
        throw ArgumentError("content", contentLevel, "Invalid content scope");

    bool htmlLike = compDesc["GenerateDot"]["html"].get<bool>();

    std::ofstream ostream;
    ostream.open(compDesc["GenerateDot"]["output"].get<std::string>(), std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError("output", compDesc["GenerateDot"]["output"].get<std::string>(), "Unable to open output file");

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    if (outputScope != "DataModel")
    {
        
        OpModel opModel(model);

        for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
        {
            
            if (!(outputScope == "ControlModel" || outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel") || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output))
            {
                string nodeDef = "\t\"" + opIt->getName() + "\" [shape=box,"; 
                
                if (htmlLike)
                {
                    nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + opIt->getName() + "</B></FONT></TD></TR>";
                    if (contentLevel == "full")
                    {   
                        allocator::vector<string> attrKeys(opIt->getAttrKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + opIt->getAttr(*attrIt).getContentStr() + "</FONT></TD></TR>";
                    }
                    else
                    {
                        nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + Printable::toString(opIt->getOpType()) + "</FONT></TD></TR>";
                    }
                    nodeDef += "</TABLE>>";
                }
                else
                {
                    nodeDef += " label=\"" + opIt->getName() + "\\n";
                    if (contentLevel == "full")
                    {   
                        allocator::vector<string> attrKeys(opIt->getAttrKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += *attrIt + ": " + opIt->getAttr(*attrIt).getContentStr() + "\\n";
                    }
                    nodeDef += "\"";
                }
                
                ostream << nodeDef << "];\n";

            }

        }

        if (outputScope == "OpModel" || outputScope == "ExecOpModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {
            
            DataModel dataModel(model);

            for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
            {
                if (!(outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel") || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output))
                {
                    for (auto dataIt = opIt.leftmostOutput(); dataIt != dataModel.flowEnd(); ++dataIt)
                    {

                        string edgeDef = "\t\"" + opIt->getName() + "\" -> \"" + dataIt.sink()->getName() + "\"";
                        if (htmlLike)
                        {
                            edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + dataIt->getTensor()->getName() + "</B></FONT></TD></TR>";
                            if (contentLevel == "full")
                            {   
                                allocator::vector<string> attrKeys(dataIt->getTensor()->getAttrKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + dataIt->getTensor()->getAttr(*attrIt).getContentStr() + "</FONT></TD></TR>";
                            }
                            else
                            {
                                edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + dataIt->getTensor()->getShape().toString() + "</FONT></TD></TR>";
                            }
                            edgeDef += "</TABLE>>];";
                        }
                        else
                        {
                            edgeDef += " [label=\"" + dataIt->getTensor()->getName() + "\\n";
                            if (contentLevel == "full")
                            {   
                                allocator::vector<string> attrKeys(dataIt->getTensor()->getAttrKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    edgeDef += *attrIt + ": " + dataIt->getTensor()->getAttr(*attrIt).getContentStr() + "\\n";
                            }
                            edgeDef += "\"];";
                        }

                        ostream << edgeDef << "\n";

                    }

                }

            }

        }

        if (outputScope == "ControlModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {

            ControlModel controlModel(model);

            for (auto opIt = controlModel.getFirst(); opIt != controlModel.opEnd(); ++opIt)
            {

                for (auto controlIt = opIt.leftmostOutput(); controlIt != controlModel.flowEnd(); ++controlIt)
                {

                    string edgeDef = "\t" + opIt->getName() + " -> " + controlIt.sink()->getName() + " [penwidth=2.0, style=dashed]";
                    ostream << edgeDef << "\n";

                }

            }

        }
    
    }
    else
    {

        DataModel dataModel(model);

        for (auto tIt = dataModel.tensorBegin(); tIt != dataModel.tensorEnd(); ++tIt)
        {

            string nodeDef = "\t\"" + tIt->getName() + "\" [shape=box,"; 
                
            if (htmlLike)
            {
                nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + tIt->getName() + "</B></FONT></TD></TR>";
                if (contentLevel == "full")
                {   
                    allocator::vector<string> attrKeys(tIt->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + tIt->getAttr(*attrIt).getContentStr() + "</FONT></TD></TR>";
                }
                else
                {
                    nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + tIt->getShape().toString() + "</FONT></TD></TR>";
                }
                nodeDef += "</TABLE>>";
            }
            else
            {
                nodeDef += " label=\"" + tIt->getName() + "\\n";
                if (contentLevel == "full")
                {   
                    allocator::vector<string> attrKeys(tIt->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += *attrIt + ": " + tIt->getAttr(*attrIt).getContentStr() + "\\n";
                }
                nodeDef += "\"";
            }
            
            ostream << nodeDef << "];\n";
        
        }

        for (auto flowIt = dataModel.flowBegin(); flowIt != dataModel.flowEnd(); ++flowIt)
        {

            if (flowIt.childrenSize() > 0)
            {
            string edgeDef = "\t\"" + flowIt->getTensor()->getName() + "\" -> \"" + flowIt.leftmostChild()->getTensor()->getName() + "\"";
            if (htmlLike)
            {
                edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + flowIt.sink()->getName() + "</B></FONT></TD></TR>";
                if (contentLevel == "full")
                {   
                    allocator::vector<string> attrKeys(flowIt.sink()->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + flowIt.sink()->getAttr(*attrIt).getContentStr() + "</FONT></TD></TR>";
                }
                else
                {
                    edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + Printable::toString(flowIt.sink()->getOpType()) + "</FONT></TD></TR>";
                }

                edgeDef += "</TABLE>>];";
            }
            else
            {
                edgeDef += " [label=\"" + flowIt.sink()->getName() + "\\n";
                if (contentLevel == "full")
                {   
                    allocator::vector<string> attrKeys(flowIt.sink()->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        edgeDef += *attrIt + ": " + flowIt.sink()->getAttr(*attrIt).getContentStr() + "\\n";
                }
                edgeDef += "\"];";
            }

            ostream << edgeDef << "\n";
            }

        }

    }

    ostream << "}\n";
    ostream.close();

}