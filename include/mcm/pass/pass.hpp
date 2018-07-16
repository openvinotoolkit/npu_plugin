#ifndef MV_PASS_PASS_HPP_
#define MV_PASS_PASS_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/base/stream/ostream.hpp"
#include <string>

namespace mv
{

    namespace pass
    {

        

        class Pass
        {
                
        protected:

            //static Logger &logger_;
            //static std::string name_;

            //virtual bool run_(ComputationModel &model) = 0;

        public:

            constexpr Pass();
            bool run(ComputationModel &model);
            //virtual ~Pass() = 0;

        };

        

    }

}

#endif // MV_PASS_PASS_HPP_