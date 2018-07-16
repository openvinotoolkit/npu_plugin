#ifndef MV_PASS_PASS_REGISTRY_HPP_
#define MV_PASS_PASS_REGISTRY_HPP_

#include <array>
#include <functional>
#include <string>
#include "include/mcm/pass/pass.hpp"

namespace mv
{

    namespace pass
    {

        class PassRegistry
        {
            

        public:

            static PassRegistry *instance();


        };

    }

}

#endif // MV_PASS_PASS_REGISTRY_HPP_
