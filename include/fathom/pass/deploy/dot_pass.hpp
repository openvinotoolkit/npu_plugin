#ifndef DOT_PASS_HPP_
#define DOT_PASS_HPP_

#include "include/fathom/pass/deploy/deploy_pass.hpp"
#include "include/fathom/computation/model/op_model.hpp"

namespace mv
{

    namespace pass 
    {

        class DotPass : public DeployPass
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
                ControlModel,
                OpControlModel
            };

        private:

            OutputScope outputScope_;
            ContentLevel contentLevel_;
            bool htmlLike_;

        public:

            DotPass(Logger &logger, OStream &ostream, OutputScope outputScope = OutputScope::OpControlModel, ContentLevel contentLevel = ContentLevel::ContentName, bool htmlLike = true) :
            DeployPass(logger, ostream),
            outputScope_(outputScope),
            contentLevel_(contentLevel),
            htmlLike_(htmlLike)
            {

            }

            bool run(ComputationModel &model)
            {
                OpModel opModel(model);
                ostream_ << "digraph G {\n\tgraph [splines=spline]\n";

                for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
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

                if (outputScope_ == OutputScope::OpModel || outputScope_ == OutputScope::OpControlModel)
                {
                    
                    DataModel dataModel(model);

                    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
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

                if (outputScope_ == OutputScope::ControlModel || outputScope_ == OutputScope::OpControlModel)
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

        };

    }

}

#endif // DOT_PASS_HPP_