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
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/op/ops_register.hpp"
#include "include/mcm/base/printable.hpp"

namespace mv
{
    
    enum class Target
    {
        ma2480,
        Unknown
    };

    class TargetDescriptor
    {

        static std::string toString(Target target);
        static Target toTarget(const std::string& str);
        static DType toDType(const std::string& str);
        static Order toOrder(const std::string& str);
        static OpType toOpType(const std::string str);

        const static unsigned jsonParserBufferLenght_ = 128;

        Target target_;
        DType globalDType_;
        Order globalOrder_;
        std::set<OpType> ops_;

        std::vector<std::string> adaptationPasses_;
        std::vector<std::string> optimizationPasses_;
        std::vector<std::string> finalizationPasses_;
        std::vector<std::string> serializationPasses_;
        std::vector<std::string> validationPasses_;

    public:

        TargetDescriptor(const std::string& filePath = "");
        bool load(const std::string& filePath);
        bool save(const std::string& filePath);
        void reset();

        void setTarget(Target target);
        void setDType(DType dType);
        void setOrder(Order order);

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

        bool defineOp(OpType op);
        bool undefineOp(OpType op);

        bool opSupported(OpType op) const;

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
        Order getOrder() const;
        DType getDType() const;

    };

}

#endif // TARGET_DESCRIPTOR_HPP_