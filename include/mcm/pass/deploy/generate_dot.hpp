#ifndef GENERATE_DOT_HPP_
#define GENERATE_DOT_HPP_

#include "include/mcm/pass/deploy_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

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
                ExecOpControlModel,
                DataModel
            };

        private:

            OutputScope outputScope_;
            ContentLevel contentLevel_;
            bool htmlLike_;

            bool run_(ComputationModel &model);

        public:

            GenerateDot(OStream &ostream, OutputScope outputScope = OutputScope::OpControlModel, 
                ContentLevel contentLevel = ContentLevel::ContentName, bool htmlLike = true);

        };

    }

}

#endif // DOT_PASS_HPP_