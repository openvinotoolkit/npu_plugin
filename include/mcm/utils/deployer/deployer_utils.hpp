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
        namespace dep_utils
        {
            enum class InputMode
            {
                ALL_ZERO,
                ALL_ONE,
                FILE,
                Unknown
            };
            Order getTensorOrder(ncTensorDescriptor_t& td)
            {
                unsigned int max = std::max(std::max(td.hStride, td.wStride), td.cStride);

                if (max == td.hStride)
                {

                    if (std::max(td.wStride, td.cStride) == td. wStride)
                        return Order("NHWC");
                    else
                        return Order("NHCW");

                }
                else if (max == td.cStride)
                {

                    if (std::max(td.wStride, td.hStride) == td.hStride)
                        return Order("NCHW");
                    else
                        return Order("NCWH");
                }
                else
                {
                    //W is major
                    if (std::max(td.hStride, td.cStride) == td.hStride)
                        return Order("NWHC");
                    else
                        return Order("NWCH");

                }
            }

            void checkFileExists(const std::string& fileName)
            {
                if (fileName.empty())
                    std::runtime_error(fileName + "is Empty");
                std::ifstream checkFile(fileName, std::ios::in | std::ios::binary);
                if (checkFile.fail())
                    std::runtime_error(fileName + " File not found!");
            }


            Tensor convertDataToTensor(Order& order, Shape& shape, unsigned short* imageData, int numberOfElements)
            {
                //Convert to Tensor
                std::vector<double> tensorData(numberOfElements);
                for (int i = 0; i < numberOfElements; i++) {
                    tensorData[i] = imageData[i];
                }
                Tensor resultTensor("result", shape, DType(DTypeType::Float16), order);
                resultTensor.populate(tensorData);

                return resultTensor;
            }

            Tensor convertDataToTensor(ncTensorDescriptor_t& tensorDescriptor, unsigned short* imageData)
            {
                //Convert to Tensor
                Shape shape({tensorDescriptor.w, tensorDescriptor.h, tensorDescriptor.c, tensorDescriptor.n});
                Order order = getTensorOrder(tensorDescriptor);

                return convertDataToTensor(order, shape, imageData, tensorDescriptor.totalSize);
            }

            Tensor getInputData(std::string inputFileName, Order& order, Shape& shape)
            {
                checkFileExists(inputFileName);
                std::ifstream inputFile(inputFileName, std::ios::in | std::ios::binary);
                std::cout<< "loading input image from: " << inputFileName << std::endl;

                int dataSize = shape.totalSize();
                std::unique_ptr<unsigned short[]> imageData(new unsigned short[dataSize]);

                if (!inputFile.read(reinterpret_cast<char *>(imageData.get()), dataSize*2))
                    std::runtime_error("generateInputData failed on read from file");

                return convertDataToTensor(order, shape, imageData.get(), dataSize);
            }

            Tensor getInputData(InputMode mode, Order& order, Shape& shape)
            {
                std::vector<unsigned short> myvector(shape.totalSize());

                if (mode == InputMode::ALL_ONE)
                {
                    std::fill_n(myvector.begin(), myvector.size(), 0x3c00);//fp32_to_fp16(1.0)
                }
                else
                {
                    std::fill_n(myvector.begin(), myvector.size(), 0);
                }
                return convertDataToTensor(order, shape, &myvector[0], shape.totalSize());
            }
        }
    }
}


#endif // DEPLOYER_UTILS_HPP