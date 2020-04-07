#ifndef MV_PASS_PASS_REGISTRY_HPP_
#define MV_PASS_PASS_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/pass/pass_entry.hpp"
#include "include/mcm/base/exception/master_error.hpp"

namespace mv
{

    class ComputationModel;
    class TargetDescriptor;

    namespace pass
    {   

        class PassRegistry : public Registry<PassRegistry, std::string, PassEntry>
        {
            

        public:

            static PassRegistry& instance();

            void run(std::string name, ComputationModel& model, TargetDescriptor& targetDescriptor, 
                Element& passDescriptor, Element& output);

        };

        #define MV_REGISTER_PASS(Name)                          \
            MV_REGISTER_ENTRY(PassRegistry, std::string, PassEntry, #Name)    \
                              
    }

}

#endif // MV_PASS_PASS_REGISTRY_HPP_
