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
#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/base/element.hpp"

namespace mv
{

    enum class Target
    {
        ma2480,
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

        static std::string toString(Target target);
        static Target toTarget(const std::string& str);
        const static unsigned jsonParserBufferLenght_ = 128;

        Target target_;
        DType globalDType_;
        std::set<std::string> ops_;
        std::set<std::string> postOps_;
        std::map<std::string, MemoryDescriptor> memoryDefs_;

        std::vector<std::string> adaptationPasses_;
        std::vector<std::string> optimizationPasses_;
        std::vector<std::string> finalizationPasses_;
        std::vector<std::string> serializationPasses_;
        std::vector<std::string> validationPasses_;


        std::map<std::string, mv::Element> serialDescriptions_;

    public:

        TargetDescriptor(const std::string& filePath = "");
        bool load(const std::string& filePath);
        bool save(const std::string& filePath);
        void reset();

        void setTarget(Target target);
        void setDType(DType dType);

        bool appendAdaptPass(const std::string& pass, int pos = -1);
        bool appendOptPass(const std::string& pass, int pos = -1);
        bool appendFinalPass(const std::string& pass, int pos = -1);
        bool appendSerialPass(const std::string& pass, int pos = -1);
        bool appendValidPass(const std::string& pass, int pos = -1);

        bool removeAdaptPass(const std::string& pass);
        bool removeOptPass(const std::string& pass);
        bool removeFinalPass(const std::string& pass);
        bool removeSerialPass(const std::string& pass);
        bool removeValidPass(const std::string& pass);

        bool defineOp(const std::string& opType);
        bool undefineOp(const std::string& opType);
        bool opSupported(const std::string& opType) const;
        bool opSupportedAsPostOp(const std::string& opType) const;


        bool defineMemory(const std::string& name, long long size, std::size_t alignment, std::size_t dataTypeSize);
        bool undefineMemory(const std::string& name);

        std::size_t adaptPassesCount() const;
        std::size_t optPassesCount() const;
        std::size_t finalPassesCount() const;
        std::size_t serialPassesCount() const;
        std::size_t validPassesCount() const;

        const std::vector<std::string>& adaptPasses() const;
        const std::vector<std::string>& optPasses() const;
        const std::vector<std::string>& finalPasses() const;
        const std::vector<std::string>& serialPasses() const;
        const std::vector<std::string>& validPasses() const;

        Target getTarget() const;
        DType getDType() const;

        mv::Element getSerialDefinition(std::string op_name, std::string platform_name) const;

        const std::map<std::string, MemoryDescriptor>& memoryDefs() const;

        std::string getLogID() const override;

    };

}

#endif // TARGET_DESCRIPTOR_HPP_
