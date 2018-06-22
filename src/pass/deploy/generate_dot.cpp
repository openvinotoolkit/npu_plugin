#include "include/mcm/pass/deploy/generate_dot.hpp"

mv::pass::GenerateDot::GenerateDot(OStream &ostream, OutputScope outputScope, ContentLevel contentLevel, bool htmlLike) :
DeployPass(ostream),
outputScope_(outputScope),
contentLevel_(contentLevel),
htmlLike_(htmlLike)
{

}

bool mv::pass::GenerateDot::run_(ComputationModel &model)
{

    ostream_ << "digraph G {\n\tgraph [splines=spline]\n";

    if (outputScope_ != OutputScope::DataModel)
    {
        
        OpModel opModel(model);

        for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
        {
            
            if (!(outputScope_ == OutputScope::ControlModel || outputScope_ == OutputScope::ExecOpModel || outputScope_ == OutputScope::ExecOpControlModel) || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output))
            {
                string nodeDef = "\t\"" + opIt->getName() + "\" [shape=box,"; 
                
                if (htmlLike_)
                {
                    nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + opIt->getName() + "</B></FONT></TD></TR>";
                    if (contentLevel_ == ContentLevel::ContentFull)
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
                    if (contentLevel_ == ContentLevel::ContentFull)
                    {   
                        allocator::vector<string> attrKeys(opIt->getAttrKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += *attrIt + ": " + opIt->getAttr(*attrIt).getContentStr() + "\\n";
                    }
                    nodeDef += "\"";
                }
                
                ostream_ << nodeDef << "];\n";

            }

        }

        if (outputScope_ == OutputScope::OpModel || outputScope_ == OutputScope::ExecOpModel || outputScope_ == OutputScope::OpControlModel || outputScope_ == OutputScope::ExecOpControlModel)
        {
            
            DataModel dataModel(model);

            for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
            {
                if (!(outputScope_ == OutputScope::ExecOpModel || outputScope_ == OutputScope::ExecOpControlModel) || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output))
                {
                    for (auto dataIt = opIt.leftmostOutput(); dataIt != dataModel.flowEnd(); ++dataIt)
                    {

                        string edgeDef = "\t\"" + opIt->getName() + "\" -> \"" + dataIt.sink()->getName() + "\"";
                        if (htmlLike_)
                        {
                            edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + dataIt->getTensor()->getName() + "</B></FONT></TD></TR>";
                            if (contentLevel_ == ContentLevel::ContentFull)
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
                            if (contentLevel_ == ContentLevel::ContentFull)
                            {   
                                allocator::vector<string> attrKeys(dataIt->getTensor()->getAttrKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    edgeDef += *attrIt + ": " + dataIt->getTensor()->getAttr(*attrIt).getContentStr() + "\\n";
                            }
                            edgeDef += "\"];";
                        }

                        ostream_ << edgeDef << "\n";

                    }

                }

            }

        }

        if (outputScope_ == OutputScope::ControlModel || outputScope_ == OutputScope::OpControlModel || outputScope_ == OutputScope::ExecOpControlModel)
        {

            ControlModel controlModel(model);

            for (auto opIt = controlModel.getFirst(); opIt != controlModel.opEnd(); ++opIt)
            {

                for (auto controlIt = opIt.leftmostOutput(); controlIt != controlModel.flowEnd(); ++controlIt)
                {

                    string edgeDef = "\t" + opIt->getName() + " -> " + controlIt.sink()->getName() + " [penwidth=2.0, style=dashed]";
                    ostream_ << edgeDef << "\n";

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
                
            if (htmlLike_)
            {
                nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + tIt->getName() + "</B></FONT></TD></TR>";
                if (contentLevel_ == ContentLevel::ContentFull)
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
                if (contentLevel_ == ContentLevel::ContentFull)
                {   
                    allocator::vector<string> attrKeys(tIt->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += *attrIt + ": " + tIt->getAttr(*attrIt).getContentStr() + "\\n";
                }
                nodeDef += "\"";
            }
            
            ostream_ << nodeDef << "];\n";
        
        }

        for (auto flowIt = dataModel.flowBegin(); flowIt != dataModel.flowEnd(); ++flowIt)
        {

            if (flowIt.childrenSize() > 0)
            {
            string edgeDef = "\t\"" + flowIt->getTensor()->getName() + "\" -> \"" + flowIt.leftmostChild()->getTensor()->getName() + "\"";
            if (htmlLike_)
            {
                edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + flowIt.sink()->getName() + "</B></FONT></TD></TR>";
                if (contentLevel_ == ContentLevel::ContentFull)
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
                if (contentLevel_ == ContentLevel::ContentFull)
                {   
                    allocator::vector<string> attrKeys(flowIt.sink()->getAttrKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        edgeDef += *attrIt + ": " + flowIt.sink()->getAttr(*attrIt).getContentStr() + "\\n";
                }
                edgeDef += "\"];";
            }

            ostream_ << edgeDef << "\n";
            }

        }

    }

    ostream_ << "}\n";

    return true;

}