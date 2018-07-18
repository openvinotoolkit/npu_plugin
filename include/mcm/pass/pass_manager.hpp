#ifndef PASS_MANAGER_HPP_
#define PASS_MANAGER_HPP_

#include <vector>
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

namespace mv
{

    class PassManager
    {

        bool ready_;
        bool completed_;

        TargetDescriptor targetDescriptor_;
        ComputationModel *model_;

        std::vector<std::string> adaptPassQueue_;
        std::vector<std::string> optPassQueue_;
        std::vector<std::string> finalPassQueue_;
        std::vector<std::string> serialPassQueue_;
        std::vector<std::string> validPassQueue_;

        const std::vector<std::pair<PassGenre, std::vector<std::string>&>> passFlow_ =
        {
            {PassGenre::Adaptation, adaptPassQueue_},
            {PassGenre::Validation, validPassQueue_},
            {PassGenre::Optimization, optPassQueue_},
            {PassGenre::Validation, validPassQueue_},
            {PassGenre::Finalization, finalPassQueue_},
            {PassGenre::Validation, validPassQueue_},
            {PassGenre::Serialization, serialPassQueue_}
        };

        std::vector<std::pair<PassGenre, std::vector<std::string>&>>::const_iterator currentStage_;
        std::vector<std::string>::iterator currentPass_;

    public:

        PassManager();
        bool initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor);
        void reset();
        bool ready() const;
        bool completed() const;
        std::pair<std::string, PassGenre> step();
    

    };

}

#endif // PASS_MANAGER_HPP_
