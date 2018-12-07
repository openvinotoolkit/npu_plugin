#include "include/mcm/utils/deployer/configuration.hpp"

namespace mv
{
    namespace exe
    {
        void Configuration::checkFileExists(const std::string& fileName, const std::string& argName)
        {
            if (fileName.empty())
                throw ArgumentError(*this, argName, "Empty", "Defining file path is illegal");
            std::ifstream checkFile(fileName, std::ios::in | std::ios::binary);
            if (checkFile.fail())
                    throw ArgumentError(*this, argName,
                        fileName, " File not found!");
        }

        Configuration::Configuration(const std::string& graphFilePath):
            target_(Target::Unknown), //unknown == any device is good.
            protocol_(Protocol::USB_VSC),
            inputMode_(InputMode::ALL_ZERO),
            binaryPointer_(nullptr)
        {
            checkFileExists(graphFilePath, "Graph File");
            graphFilePath_ = graphFilePath;
        }

        Configuration::Configuration(const std::string& graphFilePath,
            Target target, Protocol protocol,
            InputMode inputMode, const std::string& inputFilePath):
            binaryPointer_(nullptr)
        {
            checkFileExists(graphFilePath, "Graph File");
            graphFilePath_ = graphFilePath;
            setTarget(target);
            setProtocol(protocol);
            setInputMode(inputMode);
            setInputFilePath(inputFilePath);
        }
        Configuration::Configuration(std::shared_ptr<RuntimeBinary> binaryPointer):
            target_(Target::Unknown), //unknown == any device is good.
            protocol_(Protocol::USB_VSC),
            inputMode_(InputMode::ALL_ZERO),
            graphFilePath_(""),
            binaryPointer_(binaryPointer)
        {
        }

        Configuration::Configuration(std::shared_ptr<RuntimeBinary> binaryPointer,
            Target target, Protocol protocol,
            InputMode inputMode, const std::string& inputFilePath):
            graphFilePath_(""),
            binaryPointer_(binaryPointer)
        {
            setTarget(target);
            setProtocol(protocol);
            setInputMode(inputMode);
            setInputFilePath(inputFilePath);
        }

        Configuration::Configuration(const Configuration &c):
            target_(c.target_),
            protocol_(c.protocol_),
            inputMode_(c.inputMode_),
            inputFilePath_(c.inputFilePath_),
            graphFilePath_(c.graphFilePath_),
            binaryPointer_(c.binaryPointer_)
        {
        }

        void Configuration::setTarget(Target target)
        {
            target_ = target;
        }

        void Configuration::setProtocol(Protocol protocol)
        {
            if (protocol == Protocol::Unknown)
                throw ArgumentError(*this, "protocol", "unknown", "Defining protocol as unknown is illegal");
            protocol_ = protocol;
        }

        void Configuration::setInputMode(InputMode inputMode)
        {
            if (inputMode == InputMode::Unknown)
                throw ArgumentError(*this, "inputMode", "unknown", "Defining inputMode as unknown is illegal");
            inputMode_ = inputMode;
        }

        void Configuration::setInputFilePath(const std::string& inputFilePath)
        {
            checkFileExists(inputFilePath, "Input File");
            inputFilePath_ = inputFilePath;
        }

        Target Configuration::getTarget() const
        {
            return target_;
        }

        Protocol Configuration::getProtocol() const
        {
            return protocol_;
        }

        InputMode Configuration::getInputMode() const
        {
            return inputMode_;
        }

        std::string Configuration::getInputFilePath() const
        {
            return inputFilePath_;
        }

        std::string Configuration::getGraphFilePath( ) const
        {
            return graphFilePath_;
        }

        std::string Configuration::targetToString() const
        {
            switch (target_)
            {

                case Target::ma2480:
                    return "ma2480";

                default:
                    return "unknown";

            }
        }

        std::string Configuration::getLogID() const
        {
            return "Configuration" + targetToString();
        }

        std::shared_ptr<RuntimeBinary> Configuration::getRuntimePointer()
        {
            return binaryPointer_;
        }
    }
}
