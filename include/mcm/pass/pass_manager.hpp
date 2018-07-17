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

        TargetDescriptor targetDescriptor_;
        ComputationModel *model_;

        std::vector<std::string> adaptPassQueue_;
        std::vector<std::string> optPassQueue_;
        std::vector<std::string> finalPassQueue_;
        std::vector<std::string> serialPassQueue_;
        std::vector<std::string> validPassQueue_;

        PassGenre currentStage_;
        PassGenre previousStage_;
        std::vector<std::string>::iterator currentPass_;

    public:

        PassManager();
        bool initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor);
        void reset();
        bool ready();
        std::pair<std::string, PassGenre> step();
    

    };

}

#endif // PASS_MANAGER_HPP_
