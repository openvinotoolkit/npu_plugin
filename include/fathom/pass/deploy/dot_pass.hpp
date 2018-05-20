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

        private:

            ContentLevel contentLevel_;

        public:

            DotPass(Logger &logger, OStream &ostream, ContentLevel contentLevel = ContentLevel::ContentName) :
            DeployPass(logger, ostream),
            contentLevel_(contentLevel)
            {

            }

            bool run(ComputationModel &model)
            {
                OpModel opModel(model);
                ostream_ << "digraph G {\n";

                for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
                {

                    string nodeDef = "\t" + (*opIt).getName() + " [shape=box,";
                    
                    if (contentLevel_ == ContentLevel::ContentFull)
                    {
                        string nodeArgs = " label=<<B>" + (*opIt).toString();
                        nodeDef += nodeArgs.substr(0, nodeArgs.find("\n"));
                        nodeArgs.erase(0, nodeArgs.find("\n")+ 1);
                        nodeDef += "<BR/></B>" + nodeArgs;
                        Printable::replaceSub(nodeDef, "\n", "<BR/>");
                        nodeDef += ">";
                    }
                    ostream_ << nodeDef << "];\n";

                }

                for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
                {

                    for (auto dataIt = opIt->leftmost_output(); dataIt != opModel.dataEnd(); ++dataIt)
                    {

                        string edgeDef = "\t" + (*opIt).getName() + " -> " + (*(*dataIt->sink())).getName();
                        if (contentLevel_ == ContentLevel::ContentFull)
                        {
                            string edgeArgs = " [label=<<B>" + (*dataIt).getTensor().toString();
                            edgeDef += edgeArgs.substr(0, edgeArgs.find("\n"));
                            edgeArgs.erase(0, edgeArgs.find("\n")+ 1);
                            edgeDef += "<BR/></B>" + edgeArgs;
                            Printable::replaceSub(edgeDef, "\n", "<BR/>");
                            edgeDef += ">]";
                        }
                        ostream_ << edgeDef << "\n";

                    }

                }

                ostream_ << "}\n";

                return true;

            }

        };

    }

}

#endif // DOT_PASS_HPP_