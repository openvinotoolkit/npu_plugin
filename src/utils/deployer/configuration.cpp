#include "include/mcm/utils/deployer/configuration.hpp"

void mv::exe::Configuration::checkFileExists_(const std::string& fileName, const std::string& argName)
{
    if (fileName.empty())
        throw ArgumentError(*this, argName, "Empty", "Defining file path is illegal");
    std::ifstream checkFile(fileName, std::ios::in | std::ios::binary);
    if (checkFile.fail())
            throw ArgumentError(*this, argName,
                fileName, " File not found!");
}

mv::exe::Configuration::Configuration(const std::string& graphFilePath):
    target_(Target::Unknown), //unknown == any device is good.
    protocol_(Protocol::USB_VSC),
    binaryPointer_(nullptr)
{
    checkFileExists_(graphFilePath, "Graph File");
    graphFilePath_ = graphFilePath;
}

mv::exe::Configuration::Configuration(const std::string& graphFilePath,
    Target target, Protocol protocol):
    binaryPointer_(nullptr)
{
    checkFileExists_(graphFilePath, "Graph File");
    graphFilePath_ = graphFilePath;
    setTarget(target);
    setProtocol(protocol);
}

mv::exe::Configuration::Configuration(std::shared_ptr<RuntimeBinary> binaryPointer) :
    target_(Target::Unknown), //unknown == any device is good.
    protocol_(Protocol::USB_VSC),
    binaryPointer_(binaryPointer)
{

}

mv::exe::Configuration::Configuration(std::shared_ptr<RuntimeBinary> binaryPointer,
    Target target, Protocol protocol,
    InputMode inputMode, const std::string& inputFilePath) :
    graphFilePath_(""),
    binaryPointer_(binaryPointer)
{
    setTarget(target);
    setProtocol(protocol);
}

mv::exe::Configuration::Configuration(const Configuration &c) :
    target_(c.target_),
    protocol_(c.protocol_),
    graphFilePath_(c.graphFilePath_),
    binaryPointer_(c.binaryPointer_)
{

}

void mv::exe::Configuration::setTarget(Target target)
{
    target_ = target;
}

void mv::exe::Configuration::setProtocol(Protocol protocol)
{
    if (protocol == Protocol::Unknown)
        throw ArgumentError(*this, "protocol", "unknown", "Defining protocol as unknown is illegal");
    protocol_ = protocol;
}

mv::Target mv::exe::Configuration::getTarget() const
{
    return target_;
}

mv::exe::Protocol mv::exe::Configuration::getProtocol() const
{
    return protocol_;
}

std::string mv::exe::Configuration::getGraphFilePath( ) const
{
    return graphFilePath_;
}

std::string mv::exe::Configuration::getLogID() const
{
    return "Configuration:" + TargetDescriptor::toString(target_);
}

std::shared_ptr<mv::RuntimeBinary> mv::exe::Configuration::getRuntimePointer()
{
    return binaryPointer_;
}