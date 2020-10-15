#ifndef DEPLOYER_UTILS_HPP
#define DEPLOYER_UTILS_HPP

#include <vector>
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/utils/deployer/executor.hpp"
#include "mvnc.h"
namespace mv
{
    namespace exe
    {
        namespace utils
        {
            enum class InputMode
            {
                ALL_ZERO,
                ALL_ONE,
                FILE,
                Unknown
            };

            Order getTensorOrder(ncTensorDescriptor_t& td);

            void checkFileExists(const std::string& callerId, const std::string& argName, const std::string& fileName);

            Tensor convertDataToTensor(Order& order, Shape& shape, unsigned short* imageData, int numberOfElements);
            Tensor convertDataToTensor(ncTensorDescriptor_t& tensorDescriptor, unsigned short* imageData);

            Tensor getInputData(std::string inputFileName, Order& order, Shape& shape);
            Tensor getInputData(InputMode mode, Order& order, Shape& shape);
        }
    }
}


#endif // DEPLOYER_UTILS_HPP