#ifndef TRANSFORM_PASS_HPP_
#define TRANSFORM_PASS_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/logger/logger.hpp"

namespace mv
{

    namespace pass
    {

        class TransformPass
        {

        protected:

            string name_;
            static Logger &logger_;
            virtual bool run_(ComputationModel &model) = 0;

        public:

            TransformPass(const string& name);
            bool run(ComputationModel &model);
            virtual ~TransformPass() = 0;

        };

    }

}

#endif // TRANSFORM_PASS_HPP_