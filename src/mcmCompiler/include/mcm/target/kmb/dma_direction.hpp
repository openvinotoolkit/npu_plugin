#ifndef MV_DMADIRECTION_HPP_
#define MV_DMADIRECTION_HPP_

#include <string>
#include <unordered_map>
#include "include/mcm/base/exception/argument_error.hpp"

namespace mv
{
    //TODO: Add all possibile directions
    enum DmaDirectionEnum
    {
        DDR2NNCMX,
        DDR2UPACMX,
        NNCMX2DDR,
        NNCMX2UPACMX,
        UPACMX2NNCMX,
        UPACMX2DDR,
        DDR2DDR,
    };

    struct DmaDirectionEnumHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class DmaDirection : public LogSender
    {

    private:

        static const std::unordered_map<DmaDirectionEnum, std::string, DmaDirectionEnumHash> dmaDirectionStrings_;
        DmaDirectionEnum direction_;

    public:

        DmaDirection();
        DmaDirection(DmaDirectionEnum value);
        DmaDirection(const DmaDirection& other);
        DmaDirection(const std::string& value);

        std::string toString() const;

        DmaDirection& operator=(const DmaDirection& other);
        DmaDirection& operator=(const DmaDirectionEnum& other);
        operator DmaDirectionEnum() const;

        std::string getLogID() const override;

        inline friend bool operator==(const DmaDirection& a, const DmaDirection& b)
        {
            return a.direction_ == b.direction_;
        }
        inline friend bool operator==(const DmaDirection& a, const DmaDirectionEnum& b)
        {
            return a.direction_ == b;
        }
        inline friend bool operator!=(const DmaDirection& a, const DmaDirection& b)
        {
            return a.direction_ != b.direction_;
        }
        inline friend bool operator!=(const DmaDirection& a, const DmaDirectionEnum& b)
        {
            return a.direction_ != b;
        }

    };

}

#endif
