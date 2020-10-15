#ifndef MV_CONFIGURABLE_COMPILER_HPP_
#define MV_CONFIGURABLE_COMPILER_HPP_

#include "include/mcm/api/compositional_model.hpp"
#include "include/mcm/api/describable_target.hpp"
#include "include/mcm/pass/pass_manager.hpp"
#include <string>

namespace mv
{

    class ConfigurableCompiler
    {

    public:

        virtual ~ConfigurableCompiler() = 0;
        virtual bool loadConfiguration(const std::string& filePath) = 0;
        virtual bool saveConfiguration(const std::string& filePath) = 0;
        virtual void resetConfiguration() = 0;

        virtual DescribableTarget& getTargetDesc() = 0;
        virtual void resetTargetDesc() = 0;
        virtual void loadTargetDesc(const DescribableTarget& targetDesc) = 0;

        virtual CompositionalModel& getModel() = 0;
        virtual void resetModel() = 0;
        virtual void loadModel(const CompositionalModel& model) = 0;

        virtual bool enablePass(PassGenre stage, const std::string& pass, int pos = -1) = 0;
        virtual bool disablePass(PassGenre stage, const std::string& pass) = 0;
        virtual bool disablePass(PassGenre stage) = 0;
        virtual bool disablePass() = 0;

        virtual bool definePassArg(const std::string& pass, const std::string& argName, const std::string& argValue) = 0;
        virtual bool definePassArg(const std::string& pass, const std::string& argName, int argValue) = 0;
        virtual bool definePassArg(const std::string& pass, const std::string& argName, double argValue) = 0;
        virtual bool definePassArg(const std::string& pass, const std::string& argName, bool argValue) = 0;
        virtual bool definePassArg(const std::string& pass, const std::string& argName) = 0;
        virtual bool undefinePassArg(const std::string& pass, const std::string& argName) = 0;

        virtual std::size_t scheduledPassesCount(PassGenre stage) const = 0;
        virtual const std::vector<std::string>& scheduledPasses(PassGenre stage) const = 0;

        virtual bool initialize() = 0;
        virtual std::pair<std::string, mv::PassGenre> runStep() = 0;
        virtual void run() = 0;
        virtual bool completed() const = 0;

    };

}

#endif // MV_CONFIGURABLE_COMPILER_HPP_