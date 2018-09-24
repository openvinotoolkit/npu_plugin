#ifndef MV_PASS_PASS_ENTRY_HPP_
#define MV_PASS_PASS_ENTRY_HPP_

#include <string>
#include <functional>
#include <set>
#include <map>
#include "include/mcm/target/target_descriptor.hpp"

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

        class PassEntry
        {

            std::string name_;
            std::set<PassGenre> passGenre_;
            std::string description_;
            std::map<std::string, json::JSONType> requiredArgs_;
            std::function<void(ComputationModel&, TargetDescriptor&, json::Object&, json::Object&)> passFunc_;

        public:

            PassEntry(const std::string& name) :
            name_(name)
            {

            }

            inline PassEntry& setGenre(PassGenre passGenre)
            {
                assert(passGenre_.find(passGenre) == passGenre_.end() && "Duplicated pass genre definition");
                passGenre_.insert(passGenre);
                return *this;
            }

            inline PassEntry& setGenre(const std::initializer_list<PassGenre> &passGenres)
            {
                for (auto it = passGenres.begin(); it != passGenres.end(); ++it)
                    assert(passGenre_.find(*it) == passGenre_.end() && "Duplicated pass genre definition");
                passGenre_.insert(passGenres);
                return *this;
            }

            inline PassEntry& setDescription(const std::string& description)
            {
                description_ = description;
                return *this;
            }

            inline PassEntry& setFunc(const std::function<void(ComputationModel&, TargetDescriptor&, json::Object&, json::Object&)>& passFunc)
            {
                passFunc_ = passFunc;
                return *this;
            }

            inline const std::string getName() const
            {
                return name_;
            }

            inline const std::set<PassGenre> getGenre() const
            {
                return passGenre_;
            }

            inline const std::string getDescription() const
            {
                return description_;
            }

            inline PassEntry& defineArg(json::JSONType argType, std::string argName)
            {
                assert(requiredArgs_.find(argName) == requiredArgs_.end() && "Duplicated pass argument definition");
                requiredArgs_.emplace(argName, argType);
                return *this;
            }

            inline const std::map<std::string, json::JSONType>& getArgs() const
            {
                return requiredArgs_;
            }
            
            inline std::size_t argsCount() const
            {
                return requiredArgs_.size();
            }

            inline void run(ComputationModel& model, TargetDescriptor& targetDescriptor, json::Object& compDescriptor, json::Object& output) const
            {
                passFunc_(model, targetDescriptor, compDescriptor, output);
            }


        };

        

    }

}

#endif // MV_PASS_PASS_ENTRY_HPP_