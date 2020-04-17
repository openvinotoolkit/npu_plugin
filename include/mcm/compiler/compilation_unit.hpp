#ifndef MV_COMPILATION_UNIT_HPP_
#define MV_COMPILATION_UNIT_HPP_

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/compiler/compilation_descriptor.hpp"
#include "include/mcm/compiler/compilation_profiler.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"

namespace mv
{

    class CompilationUnit : public LogSender
    {

        static const std::string ma2490DefTargetDescPath_;

        static const std::string ma2490DefCompDescPath_;
        static const std::string compositionalModelRecordingsPath_;

        static Logger& logger_;

        OpModel* model_;
        PassManager passManager_;
        TargetDescriptor targetDescriptor_;
        CompilationDescriptor compDescriptor_;
        const static unsigned jsonParserBufferLength_ = 256;

    public:
        std::shared_ptr<std::vector<char> > getBlob() const;

        CompilationUnit(const std::string& modelName);
        ~CompilationUnit();

        bool loadTargetDescriptor(const std::string& path);
        bool loadTargetDescriptor(Target target);
        bool loadCompilationDescriptor(const std::string& path);
        bool loadCompilationDescriptor(Target target);

        CompilationDescriptor& compilationDescriptor();
        OpModel& model();

        bool initialize();
        Element runStep();
        Element run();
        bool completed() const;

        virtual std::string getLogID() const override;

    };

}

#endif // MV_COMPILATION_UNIT_HPP_
