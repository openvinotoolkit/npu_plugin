#ifndef MV_PASS_PASS_ENTRY_HPP_
#define MV_PASS_PASS_ENTRY_HPP_

#include <string>
#include <functional>
#include <set>
#include <map>
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/exception/master_error.hpp"

namespace mv
{

    class ComputationModel;

    namespace pass
    {

        class PassEntry : public LogSender
        {

            std::string name_;
            std::string description_;
            std::map<std::string, json::JSONType> requiredArgs_;
            std::set<std::string> labels_;
            std::function<void(const PassEntry& pass, ComputationModel&, TargetDescriptor&, Element&, Element&)> passFunc_;

        public:

            PassEntry(const std::string& name);
            PassEntry& setDescription(const std::string& description);
            PassEntry& setFunc(const std::function<void(const PassEntry&, ComputationModel&, TargetDescriptor&, 
                Element&, Element&)>& passFunc);
            PassEntry& defineArg(json::JSONType argType, std::string argName);
            PassEntry& setLabel(const std::string& label);

            const std::string getName() const;
            const std::string getDescription() const;
            const std::map<std::string, json::JSONType>& getArgs() const;
            std::size_t argsCount() const;
            bool hasLabel(const std::string& label) const;

            void run(ComputationModel& model, TargetDescriptor& targetDescriptor, Element& passDescriptor, Element& output) const;
            
            std::string getLogID() const override;


        };

        

    }

}

#endif // MV_PASS_PASS_ENTRY_HPP_