#include "include/mcm/tensor/binarydata.hpp"

mv::BinaryData::BinaryData(mv::DTypeType type) : type_(type), data_{nullptr}
{

}
mv::BinaryData::~BinaryData()
{
    switch(type_) {
        case mv::DTypeType::Float16:
            if (data_.fp16 != nullptr)
                delete data_.fp16;
            break;
        default:
            break;
    }
}