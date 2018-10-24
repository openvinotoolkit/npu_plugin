#ifndef MV_PASS_PASS_REGISTRY_HPP_
#define MV_PASS_PASS_REGISTRY_HPP_

#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/pass/pass_entry.hpp"

namespace mv
{

    class ComputationModel;
    class TargetDescriptor;

    namespace pass
    {   

        class PassRegistry : public Registry<std::string, PassEntry>
        {
            

        public:

            static PassRegistry& instance();

            inline void run(std::string name, ComputationModel& model, TargetDescriptor& targetDescriptor, json::Object& compDescriptor, json::Object& output)
            {   
                PassEntry* const passPtr = find(name);
                if (passPtr)
                {
                    passPtr->run(model, targetDescriptor, compDescriptor, output);
                }
            }

        };

        #define MV_REGISTER_PASS(Name)                          \
            MV_REGISTER_ENTRY(std::string, PassEntry, #Name)    \
                              
    }

}

#endif // MV_PASS_PASS_REGISTRY_HPP_
