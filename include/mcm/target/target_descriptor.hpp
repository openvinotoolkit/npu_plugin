#ifndef TARGET_DESCRIPTOR_HPP_
#define TARGET_DESCRIPTOR_HPP_

#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <algorithm>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/parser/json_text.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/base/element.hpp"

namespace mv
{

    enum class Target
    {
        ma2480,
        ma2490,
        Unknown
    };

    class TargetDescriptor : public LogSender
    {

        struct MemoryDescriptor
        {

            long long size;
            std::size_t alignment;
            std::size_t dataTypeSize;

        };

        static Target toTarget(const std::string& str);
        const static unsigned jsonParserBufferLenght_ = 128;

        Target target_;
        DType globalDType_;
        std::set<std::string> ops_;
        std::set<std::string> postOps_;
        std::map<std::string, MemoryDescriptor> memoryDefs_;
        std::map<std::string, mv::Element> serialDescriptions_;

    public:

        TargetDescriptor(const std::string& filePath = "");
        bool load(const std::string& filePath);
        bool save(const std::string& filePath);
        void reset();

        void setTarget(Target target);
        void setDType(DType dType);

        bool defineOp(const std::string& opType);
        bool undefineOp(const std::string& opType);
        bool opSupported(const std::string& opType) const;
        bool opSupportedAsPostOp(const std::string& opType) const;


        bool defineMemory(const std::string& name, long long size, std::size_t alignment, std::size_t dataTypeSize);
        bool undefineMemory(const std::string& name);

        Target getTarget() const;
        DType getDType() const;

        mv::Element getSerialDefinition(std::string op_name, std::string platform_name) const;

        const std::map<std::string, MemoryDescriptor>& memoryDefs() const;

        std::string getLogID() const override;

        static std::string toString(Target target);

    };

}

#endif // TARGET_DESCRIPTOR_HPP_
