#ifndef CONFIGURATION_HPP_
#define CONFIGURATION_HPP_

#include "include/mcm/base/printable.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"


namespace mv
{
    namespace exe
    {

        enum class Protocol
        {
            USB_VSC,
            Unknown
        };

        enum class InputMode
        {
            ALL_ZERO,
            ALL_ONE,
            FILE,
            Unknown
        };

        class Configuration : public LogSender
        {

            Target target_;
            Protocol protocol_;
            std::string graphFilePath_;
            std::shared_ptr<mv::RuntimeBinary> binaryPointer_;

            void checkFileExists_(const std::string& fileName, const std::string& argName);

        public:

            Configuration(const std::string& graphFilePath);
            Configuration(const std::string& graphFilePath,
                Target target, Protocol protocol);
            Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer);
            Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer,
                Target target, Protocol protocol,
                InputMode inputMode, const std::string& inputFilePath);
            Configuration(const Configuration &c);

            void setTarget(Target target);
            void setProtocol(Protocol protocol);

            Target getTarget() const;
            Protocol getProtocol() const;
            std::string getGraphFilePath( ) const;
            std::string getLogID() const override;
            std::shared_ptr<mv::RuntimeBinary> getRuntimePointer();
        };

    }

}

#endif // CONFIGURATION_HPP_
