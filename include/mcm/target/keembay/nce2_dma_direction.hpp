#ifndef MV_NCE2_DMADIRECTION_HPP_
#define MV_NCE2_DMADIRECTION_HPP_

#include <string>
#include <unordered_map>
#include "include/mcm/base/exception/argument_error.hpp"

namespace mv
{
    //TODO: Add all possibile directions
    enum DmaDirectionEnum
    {
        DDR2CMX,
        CMX2DDR
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
        bool operator==(const DmaDirection& other) const;
        bool operator==(const DmaDirectionEnum& other) const;
        bool operator!=(const DmaDirection& other) const;
        bool operator!=(const DmaDirectionEnum& other) const;
        operator DmaDirectionEnum() const;

        std::string getLogID() const override;

    };

}

#endif
