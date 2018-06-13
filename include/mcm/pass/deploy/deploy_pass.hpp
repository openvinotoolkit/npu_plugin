#ifndef DEPLOY_PASS_HPP_
#define DEPLOY_PASS_HPP_

#include "include/mcm/computation/model/model.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/deployer/ostream.hpp"

namespace mv
{

    namespace pass
    {

        class DeployPass
        {

        protected:

            Logger &logger_;
            OStream &ostream_;

            virtual bool run_(ComputationModel &model) = 0;

        public:

            DeployPass(Logger &logger, OStream &ostream);
            bool run(ComputationModel &model);
            virtual ~DeployPass() = 0;

        };

    }

}

#endif // DEPLOY_PASS_HPP_