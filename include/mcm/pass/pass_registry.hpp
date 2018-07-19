#ifndef MV_PASS_PASS_REGISTRY_HPP_
#define MV_PASS_PASS_REGISTRY_HPP_

#include "include/mcm/base/registry.hpp"
#include "include/mcm/pass/pass_entry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(PassEntry)

    class ComputationModel;
    class TargetDescriptor;

    namespace pass
    {

        class PassRegistry : public Registry<PassEntry>
        {
            

        public:

            static PassRegistry& instance();

            inline void run(std::string name, ComputationModel& model, TargetDescriptor& targetDescriptor, json::Object& compDescriptor)
            {   
                PassEntry* const passPtr = find(name);
                if (passPtr)
                {
                    passPtr->run(model, targetDescriptor, compDescriptor);
                }
            }

        };

        #define MV_REGISTER_PASS(Name)          \
            MV_REGISTER_ENTRY(PassEntry, Name)  
                              
    }

}

#endif // MV_PASS_PASS_REGISTRY_HPP_
