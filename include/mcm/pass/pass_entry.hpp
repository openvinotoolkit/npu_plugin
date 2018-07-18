#ifndef MV_PASS_PASS_HPP_
#define MV_PASS_PASS_HPP_

#include <string>
#include <functional>
#include <set>
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
            std::function<void(ComputationModel&, TargetDescriptor&)> passFunc_;

        public:

            PassEntry(const std::string& name) :
            name_(name)
            {

            }

            inline PassEntry& setGenre(PassGenre passGenre)
            {
                passGenre_.insert(passGenre);
                return *this;
            }

            inline PassEntry& setGenre(const std::initializer_list<PassGenre> &passGenres)
            {
                passGenre_.insert(passGenres);
                return *this;
            }

            inline PassEntry& setDescription(const std::string& description)
            {
                description_ = description;
                return *this;
            }

            inline PassEntry& setFunc(const std::function<void(ComputationModel&, TargetDescriptor&)>& passFunc)
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

            inline void run(ComputationModel& model, TargetDescriptor& descriptor) const
            {
                passFunc_(model, descriptor);
            }


        };

        

    }

}

#endif // MV_PASS_PASS_HPP_