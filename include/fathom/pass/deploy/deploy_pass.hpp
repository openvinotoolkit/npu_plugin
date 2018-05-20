#ifndef DEPLOY_PASS_HPP_
#define DEPLOY_PASS_HPP_

#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/deployer/ostream.hpp"

namespace mv
{

    namespace pass
    {

        class DeployPass
        {

        protected:

            Logger &logger_;
            OStream &ostream_;

        public:

            DeployPass(Logger &logger, OStream &ostream);
            virtual bool run(ComputationModel &model) = 0;
            virtual ~DeployPass() = 0;

        };

    }

}

#endif // DEPLOY_PASS_HPP_