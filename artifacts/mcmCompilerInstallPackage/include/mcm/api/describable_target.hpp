#ifndef MV_DESCRIBABLE_TARGET_HPP_
#define MV_DESCRIBABLE_TARGET_HPP_

#include "include/mcm/target/target_descriptor.hpp"
#include <string>

namespace mv
{

    class DescribableTarget
    {

    public:

        virtual ~DescribableTarget() = 0;
        virtual bool loadDescriptor(const std::string& filePath) = 0;
        virtual bool saveDescriptor(const std::string& filePath) = 0;

        virtual void setTarget(Target target) = 0;
        virtual void setDType(DType dType) = 0;
        virtual void setOrder(Order order) = 0;

        virtual Target getTarget() const = 0;
        virtual Order getOrder() const = 0;
        virtual DType getDType() const = 0;

        virtual bool defineOp(OpType op) = 0;
        virtual bool undefineOp(OpType op) = 0;
        virtual std::vector<OpType> getOps() const = 0;
        virtual bool opSupported(OpType op) const = 0;

        virtual bool defineAllocator(const std::string& allocator, std::size_t size) = 0;
        virtual bool undefineAllocator(const std::string& allocator) = 0;
        virtual const std::vector<std::string>& getAllocators() const = 0;

        virtual bool defineCompResource(const std::string& resouce, std::size_t quantity) = 0;
        virtual bool undefineCompResource(const std::string& resouce) = 0;
        virtual bool defineCompResourceArg(const std::string& argName, DType argValue) = 0;
        virtual bool defineCompResourceArg(const std::string& argName, Order argValue) = 0;
        virtual bool defineCompResourceArg(const std::string& argName, const std::string& argValue) = 0;
        virtual bool defineCompResourceArg(const std::string& argName, double argValue) = 0;
        virtual bool defineCompResourceArg(const std::string& argName, int argValue) = 0;
        virtual bool undefineCompResourceArg(const std::string& argName) = 0;
        virtual bool addCompResourceOp(const std::string& resource, OpType op) = 0;
        virtual bool removeCompResourceOp(const std::string& resouce, OpType op) = 0;
        virtual std::size_t getCompResourceQuantity(const std::string& resouce) const = 0;
        virtual std::vector<OpType> getCompResourceOps(const std::string& resouce) const = 0;
        virtual const std::vector<std::string>& getCompResources() const = 0;

        virtual bool appendAdaptPass(const std::string& pass, int pos = -1) = 0;
        virtual bool appendOptPass(const std::string& pass, int pos = -1) = 0;
        virtual bool appendFinalPass(const std::string& pass, int pos = -1) = 0;
        virtual bool appendSerialPass(const std::string& pass, int pos = -1) = 0;
        virtual bool appendValidPass(const std::string& pass, int pos = -1) = 0;
        virtual bool removeAdaptPass(const std::string& pass) = 0;
        virtual bool removeOptPass(const std::string& pass) = 0;
        virtual bool removeFinalPass(const std::string& pass) = 0;
        virtual bool removeSerialPass(const std::string& pass) = 0;
        virtual bool removeValidPass(const std::string& pass) = 0;
 
        virtual const std::vector<std::string>& getAdaptPasses() const = 0;
        virtual const std::vector<std::string>& getOptPasses() const = 0;
        virtual const std::vector<std::string>& getFinalPasses() const = 0;
        virtual const std::vector<std::string>& getSerialPasses() const = 0;
        virtual const std::vector<std::string>& getValidPasses() const = 0;
        virtual std::size_t adaptPassesCount() const = 0;
        virtual std::size_t optPassesCount() const = 0;
        virtual std::size_t finalPassesCount() const = 0;
        virtual std::size_t serialPassesCount() const = 0;
        virtual std::size_t validPassesCount() const = 0;

    };

}

#endif // MV_DESCRIBABLE_TARGET_HPP_
