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

    enum class PassGenre
    {

        Adaptation,
        Optimization,
        Finalization,
        Serialization,
        Validation

    };

    namespace pass
    {

        class PassEntry : public LogSender
        {

            std::string name_;
            std::set<PassGenre> passGenre_;
            std::string description_;
            std::map<std::string, json::JSONType> requiredArgs_;
            std::function<void(const PassEntry& pass, ComputationModel&, TargetDescriptor&, json::Object&, json::Object&)> passFunc_;

        public:

            PassEntry(const std::string& name);
            PassEntry& setGenre(PassGenre passGenre);
            PassEntry& setGenre(const std::initializer_list<PassGenre> &passGenres);
            PassEntry& setDescription(const std::string& description);
            PassEntry& setFunc(const std::function<void(const PassEntry&, ComputationModel&, TargetDescriptor&, 
                json::Object&, json::Object&)>& passFunc);
            PassEntry& defineArg(json::JSONType argType, std::string argName);

            const std::string getName() const;
            const std::set<PassGenre> getGenre() const;
            const std::string getDescription() const;
            const std::map<std::string, json::JSONType>& getArgs() const;
            std::size_t argsCount() const;

            void run(ComputationModel& model, TargetDescriptor& targetDescriptor, json::Object& compDescriptor, json::Object& output) const;
            
            std::string getLogID() const override;


        };

        

    }

}

#endif // MV_PASS_PASS_ENTRY_HPP_