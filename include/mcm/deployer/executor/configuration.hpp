#ifndef CONFIGURATION_HPP_
#define CONFIGURATION_HPP_

#include "include/mcm/base/printable.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"


namespace mv
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
        mv::Target target_;
        Protocol protocol_;
        InputMode inputMode_;
        std::string inputFilePath_;
        std::string graphFilePath_;
        std::shared_ptr<mv::RuntimeBinary> binaryPointer_;

        std::string targetToString() const;

    public:
        Configuration(std::string& graphFilePath);
        Configuration(std::string& graphFilePath,
            Target target, Protocol protocol,
            InputMode inputMode, const std::string& inputFilePath);
        Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer);
        Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer,
            Target target, Protocol protocol,
            InputMode inputMode, const std::string& inputFilePath);
        Configuration(const Configuration &c);
        void setTarget(Target target);
        void setProtocol(Protocol protocol);
        void setInputMode(InputMode inputMode);
        void setInputFilePath(const std::string& inputFilePath);

        Target getTarget() const;
        Protocol getProtocol() const;
        InputMode getInputMode() const;
        std::string getInputFilePath( ) const;
        std::string getGraphFilePath( ) const;
        std::string getLogID() const override;
        std::shared_ptr<mv::RuntimeBinary> getRuntimePointer() const;
    };

}

#endif // CONFIGURATION_HPP_
