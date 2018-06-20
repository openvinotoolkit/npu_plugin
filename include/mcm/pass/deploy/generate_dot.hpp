#ifndef GENERATE_DOT_HPP_
#define GENERATE_DOT_HPP_

#include "include/mcm/pass/deploy_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"

namespace mv
{

    namespace pass 
    {

        class GenerateDot : public DeployPass
        {
        
        public:

            enum class ContentLevel
            {
                ContentName,
                ContentFull
            };

            enum class OutputScope
            {
                OpModel,
                ExecOpModel,
                ControlModel,
                OpControlModel,
                ExecOpControlModel
            };

        private:

            OutputScope outputScope_;
            ContentLevel contentLevel_;
            bool htmlLike_;

            bool run_(ComputationModel &model)
            {
                OpModel opModel(model);
                ostream_ << "digraph G {\n\tgraph [splines=spline]\n";

                for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
                {
                    
                    if (!(outputScope_ == OutputScope::ControlModel || outputScope_ == OutputScope::ExecOpModel || outputScope_ == OutputScope::ExecOpControlModel) || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output))
                    {
                        string nodeDef = "\t" + opIt->getName() + " [shape=box,"; 
                        
                        if (htmlLike_)
                        {
                            nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + opIt->getName() + "</B></FONT></TD></TR>";
                            if (contentLevel_ == ContentLevel::ContentFull)
                            {   
                                allocator::vector<string> attrKeys(opIt->getAttrKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + opIt->getAttr(*attrIt).getContentStr() + "</FONT></TD></TR>";
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

                                string edgeDef = "\t" + opIt->getName() + " -> " + dataIt.sink()->getName();
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

                ostream_ << "}\n";

                return true;

            }

        public:

            GenerateDot(OStream &ostream, OutputScope outputScope = OutputScope::OpControlModel, ContentLevel contentLevel = ContentLevel::ContentName, bool htmlLike = true) :
            DeployPass(ostream),
            outputScope_(outputScope),
            contentLevel_(contentLevel),
            htmlLike_(htmlLike)
            {

            }

        };

    }

}

#endif // DOT_PASS_HPP_