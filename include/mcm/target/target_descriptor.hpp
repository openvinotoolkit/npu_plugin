#ifndef TARGET_DESCRIPTOR_HPP_
#define TARGET_DESCRIPTOR_HPP_

#include <vector>
#include <string>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/parser/json_text.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/computation/model/types.hpp"

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
        static std::string toString(DType dType);
        static std::string toString(Order order);
        
        static Target toTarget(const std::string& str);
        static DType toDType(const std::string& str);
        static Order toOrder(const std::string& str);

        const static unsigned jsonParserBufferLenght_ = 128;

        Target target_;
        DType globalDType_;
        Order globalOrder_;

        std::vector<std::string> adaptationPasses_;
        std::vector<std::string> optimizationPasses_;
        std::vector<std::string> finalizationPasses_;
        std::vector<std::string> serializationPasses_;
        std::vector<std::string> validationPasses_;

    public:

        TargetDescriptor();
        bool load(const std::string& filePath);
        bool save(const std::string& filePath);

        const std::vector<std::string>& adaptationPasses() const;
        const std::vector<std::string>& optimizationPasses() const;
        const std::vector<std::string>& finalizationPasses() const;
        const std::vector<std::string>& serializationPasses() const;
        const std::vector<std::string>& validationPasses() const;

        Order getOrder() const;
        DType getDType() const;


    };

}

#endif // TARGET_DESCRIPTOR_HPP_