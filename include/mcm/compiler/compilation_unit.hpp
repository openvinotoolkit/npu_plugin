#ifndef MV_COMPILATION_UNIT_HPP_
#define MV_COMPILATION_UNIT_HPP_

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/utils/env_loader.hpp"

namespace mv
{

    class CompilationUnit
    {

        static const std::string ma2480DefDescPath_;

        OpModel* model_;
        PassManager passManager_;
        TargetDescriptor targetDescriptor_;
        json::Object compilationDescriptor_;

    public:

        CompilationUnit(mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseSilent, bool logTime = false);
        ~CompilationUnit();
        
        bool loadTargetDescriptor(const std::string& path);
        bool loadTargetDescriptor(Target target);

        PassManager& passManager();
        json::Object& compilationDescriptor();
        CompositionalModel& model();

        bool initialize();
        json::Object runStep();
        json::Object run();
        bool completed() const;

    };

}

#endif // MV_COMPILATION_UNIT_HPP_