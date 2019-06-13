#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <memory>
#include <algorithm>
#include <vector>
#include <iterator>
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/data_element.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/value_error.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/target/keembay/workload_struct.hpp"

namespace mv
{

    class Tensor : public Element
    {
    private:
        std::vector<DataElement> data_;

        std::size_t blockSize_;
        std::vector<std::vector<DataElement>::iterator> blocks_;

        Shape shape_;
        Order internalOrder_;
        std::shared_ptr<Tensor> sparsityMap_;
        std::shared_ptr<Tensor> storageElement_;
        std::vector<std::shared_ptr<Tensor>> subTensors_;
        size_t noneZeroElements_;

        bool elementWiseChecks_(const Tensor& other);
        void elementWiseDouble_(const Tensor& other, const std::function<double(double, double)>& opFunc);
        void elementWiseInt_(const Tensor& other, const std::function<int64_t(int64_t, int64_t)>& opFunc);


        std::vector<std::size_t> indToSub_(const Shape& s, unsigned index) const;
        unsigned subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const;
        void populateSparsityMapTensor_();
        void setSubtensorsOrder_(Order order);

    public:
        //NOTE: Is this method operating on I/O tensors, Weight tensors or both
        std::vector<int64_t> getZeroPointsPerChannel();

        Tensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const mv::QuantizationParams& quantParams);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const mv::QuantizationParams& quantParams, bool flag);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data, const mv::QuantizationParams& quantParams);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<int64_t>& data);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<int64_t>& data, const mv::QuantizationParams& quantParams);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<mv::DataElement>& data);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<mv::DataElement>& data, const mv::QuantizationParams& quantParams);
        Tensor(const Tensor& other);
        ~Tensor();

        void populate(const std::vector<double>& data);
        void populate(const std::vector<double>& data, Order order);
        void populate(const std::vector<int64_t>& data);
        void populate(const std::vector<int64_t>& data, Order order);
        void populate(const std::vector<mv::DataElement>& data);
        void populate(const std::vector<mv::DataElement>& data, Order order);

        void unpopulate();

        void setSparse();
        /**
         * @brief Binds the data (values vector) of this tensor (slave) to the given master tensor. After this operation data accessed
         * from this tensor will be actually read/written to the master tensor. Using the leftPadding and rightPadding it is possible
         * to select a fragment of the master tensor. Shape of the calling tensor will be modified according to the shape of master tensor
         * and padding values. Data type and data order will be inherited from the master tensor. Automatically sets populated flag.
         * Current implementation will disallow any further reordering (setOrder()) and broadcasting (broadcast()) of both master and slave.
         * @param other Master tensor, must be populated
         * @param leftPadding Vector of values specifing the padding between the bounderies (left-top) of the master tensor and this tensor per dimenision.
         * @param rightPadding Vector of values specifing the padding between the bounderies (right-bottom) of the master tensor and this tensor per dimenision.
         */
        void bindData(Tensor& other, const std::vector<std::size_t>& leftPadding = {}, const std::vector<std::size_t>& rightPadding = {});
        void broadcast(const Shape& shape);

        inline bool isDoubleType() const {
            return getDType().isDoubleType();
        }

        std::vector<DataElement> getData();
        std::vector<DataElement> getDataPacked();
        std::vector<double> getDoubleData();
        std::vector<int64_t> getIntData();
        void setDType(DType dType);
        DType getDType() const;
        void setOrder(Order order, bool updateSubtensors = false);
        Order getOrder() const;
        const Order& getInternalOrder() const;
        void setShape(const Shape& shape);
        void setAddress(int64_t address);

        void add(const Tensor& other);
        void add(double val);
        void subtract(const Tensor& other);
        void subtract(double val);
        void multiply(const Tensor& other);
        void multiply(double val);
        void divide(const Tensor& other);
        void divide(double val);
        void sqrt();

        int computeMemoryRequirement() const;

        DataElement& at(const std::vector<std::size_t>& sub);
        const DataElement& at(const std::vector<std::size_t>& sub) const;
        DataElement& at(std::size_t idx);
        const DataElement& at(std::size_t idx) const;
        DataElement& operator()(std::size_t idx);
        const DataElement& operator()(std::size_t idx) const;
        DataElement& operator()(const std::vector<std::size_t>& sub);
        const DataElement& operator()(const std::vector<std::size_t>& sub) const;

        inline bool isQuantized() const
        {
            return hasAttr("quantParams") &&
                !(getDType() == DType("Float16") || getDType() == DType("Float32") || getDType() == DType("Float64"));
        }

        inline bool isPopulated() const
        {
            return get<bool>("populated");
        }

        inline bool isSparse() const
        {
            if (hasAttr("sparse"))
                return get<bool>("sparse");
            return false;
        }

        inline bool isBroadcasted() const
        {
            if (hasAttr("broadcasted"))
                return get<bool>("broadcasted");
            return true; //by default is true
        }
        inline Shape& getShape()
        {
            return shape_;
        }

        inline const Shape& getShape() const
        {
            return shape_;
        }

        inline unsigned size() const
        {
            return shape_.totalSize();
        }

        inline unsigned sizeBytes() const
        {
            return shape_.totalSize() * (getDType().getSizeInBits()/8);
        }

        inline std::vector<std::size_t> indToSub(unsigned index) const
        {
            return indToSub_(getShape(), index);
        }

        inline unsigned subToInd(const std::vector<std::size_t>& sub) const
        {
            return subToInd_(getShape(), sub);
        }
        inline int64_t getAddress() const
        {
            return get<int64_t>("address");
        }

        std::shared_ptr<Tensor> getSparsityMap() const;
        std::shared_ptr<Tensor> getStorageElement() const;
        const Tensor& getSubTensor(uint8_t cluster);
        const Tensor& broadcastSubtensor(uint8_t cluster);

        Tensor& operator=(const Tensor& other);

        std::string toString() const override;
        virtual std::string getLogID() const override;

        BinaryData toBinary();
        std::vector<unsigned> computeNumericStrides() const;
        std::size_t computeTotalSize(unsigned int alignment = 16, bool base = false) const;
        std::size_t getClusterSize(bool base = false) const;
        void splitAcrossClusters(std::vector<Workload>, bool splitOverH, bool multicast);

    };

}

#endif // TENSOR_HPP_
