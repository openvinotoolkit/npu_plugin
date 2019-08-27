#ifndef PASS_MANAGER_HPP_
#define PASS_MANAGER_HPP_

#include <vector>
#include <algorithm>
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/compiler/compilation_descriptor.hpp"

namespace mv
{

    class PassManager : public LogSender
    {

        bool initialized_;
        bool completed_;
        bool running_;

        TargetDescriptor targetDescriptor_;
        CompilationDescriptor compDescriptor_;
        ComputationModel *model_;

        std::vector<Element> passList_;
        std::vector<Element>::const_iterator currentPass_;

        Element compOutput_;

    public:

        PassManager();
        bool initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor, const mv::CompilationDescriptor& compDescriptor);
        void reset();
        bool validDescriptors() const;
        bool initialized() const;
        bool completed() const;
        bool validPassArgs() const;
        Element& step();
        std::string getLogID() const override;

        void loadPassList(const std::vector<Element>& passList);

    };

}

#endif // PASS_MANAGER_HPP_
