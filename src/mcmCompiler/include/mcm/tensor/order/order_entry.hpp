#ifndef MV_ORDER_ENTRY
#define MV_ORDER_ENTRY

#include <vector>

namespace mv
{

    class OrderEntry
    {

        std::string name_;
        std::vector<std::size_t> contVector_;

    public:

        OrderEntry(const std::string& name) :
        name_(name)
        {

        }

        inline OrderEntry& setContiguityVector(const std::vector<size_t>& vector)
        {
            contVector_ = vector;
            return *this;
        }

        inline std::vector<size_t>& getContiguityVector()
        {
            return contVector_;
        }


    };

}

#endif // MV_ORDER_ENTRY
