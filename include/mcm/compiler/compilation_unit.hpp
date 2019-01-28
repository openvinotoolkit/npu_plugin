#ifndef MV_COMPILATION_UNIT_HPP_
#define MV_COMPILATION_UNIT_HPP_

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "meta/include/mcm/recorded_compositional_model.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/utils/env_loader.hpp"

namespace mv
{

    class CompilationUnit : public LogSender
    {

        static const std::string ma2480DefDescPath_;
        static const std::string ma2490DefDescPath_;

        static const std::string compilationDescPath_;
        static const std::string compositionalModelRecordingsPath_;

        static Logger& logger_;

        OpModel* model_;
        RecordedCompositionalModel* recordedModel_;
        PassManager passManager_;
        TargetDescriptor targetDescriptor_;
        json::Object compilationDescriptor_;
        const static unsigned jsonParserBufferLenght_ = 256;

    public:

        CompilationUnit(const std::string& modelName);
        ~CompilationUnit();
        
        bool loadTargetDescriptor(const std::string& path);
        bool loadTargetDescriptor(Target target);
        bool loadCompilationDescriptor(const std::string& path);

        PassManager& passManager();
        json::Object& compilationDescriptor();
        OpModel& model();
        CompositionalModel& recordedModel();

        void loadModelFromJson(const std::string& path);
        bool initialize();
        json::Object runStep();
        json::Object run();
        bool completed() const;

        virtual std::string getLogID() const override;

    };

}

#endif // MV_COMPILATION_UNIT_HPP_
