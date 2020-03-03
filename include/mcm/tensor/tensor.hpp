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
#include "include/mcm/target/kmb/workload_struct.hpp"

namespace mv
{

    class Tensor : public Element
    {
    public:
        class MemoryLocation  : public LogSender {
        //TODO :: find better place for it !!
        public:
            enum Location {
                NNCMX = 0,
                UPACMX =1,
                DDR = 2,
                INPUT = 3,
                OUTPUT = 4,
                BLOB = 5,
                VIRTUAL = 6,
                DEFAULT = 7
            };
        private:
            Location location_;
            bool forced_;
            //bool relocatable need this?

            static std::map<std::string,Location> createNamingMap() {
                    return {{"NNCMX",NNCMX},
                            {"UPACMX",UPACMX},
                            {"DDR",DDR},
                            {"INPUT",INPUT},
                            {"OUTPUT",OUTPUT},
                            {"BLOB",BLOB},
                            {"VIRTUAL",VIRTUAL},
                            {"DEFAULT",DEFAULT}
                    };
            }
            static std::map<std::string,Location> namingMap;

        public:
            MemoryLocation(const std::string& location) : location_(namingMap[location]),forced_(false) {}
            MemoryLocation(const Location location) : location_(location),forced_(false) {}
            MemoryLocation() : location_(DEFAULT),forced_(false) {}

            MemoryLocation(const std::string& location, bool forced) : location_(namingMap[location]),forced_(forced) {}
            MemoryLocation(const Location location, bool forced) : location_(location),forced_(forced) {}

//            MemoryLocation(MemoryLocation& location) = delete;
            void operator=(const MemoryLocation& location) = delete;
            void operator=(MemoryLocation& location) = delete;

            bool operator==(const Location other) { return (location_ == other); }
            bool operator==(std::string& other) { return (location_ == namingMap[other]);}
            bool operator==(const MemoryLocation& other) { return (location_ == other.location_);}

            bool operator!=(const Location other) {return  (location_ != other); }
            bool operator!=(std::string& other) {return ( location_ != namingMap[other]);}
            bool operator!=(const MemoryLocation& other) { return (location_ != other.location_);}

            void force() { forced_ = true;}
            bool isDefault() { return (location_ == DEFAULT); }
            bool isForced() {return forced_;}

//            void set(std::string &location) { location_ = namingMap[location]; }
//            void set(const Location location) { location_ = location; }
//            void set(const MemoryLocation& location) { location_ = location.location_; };
            bool relocate(Location newPlace)
            {
                if(forced_)
                    return false;
                else
                {
                    location_ = newPlace;
                    return true;
                }
            }

            bool relocate(std::string& newPlace)
            {
                return relocate( namingMap[newPlace]);
            }

            bool relocate(MemoryLocation& newPlace)
            {
                return relocate(newPlace.location_);
            }

            std::string toString() const
            {
                for( auto it = namingMap.begin(); it != namingMap.end(); ++it )
                {
                    if(it->second == location_)
                        return it->first;
                }
                throw ValueError(*this, "Memory location cannot be found in Map!!");
            }

            virtual std::string getLogID() const override
            {
                return "MemoryLocation";
            }
        };


    private:
        Shape shape_;
        Order internalOrder_;

        std::shared_ptr<std::vector<DataElement>> data_;

        std::size_t blockSize_;
        std::vector<std::vector<DataElement>::iterator> blocks_;

        std::shared_ptr<Tensor> sparsityMap_;
        std::shared_ptr<Tensor> storageElement_;
        std::vector<std::shared_ptr<Tensor>> subTensors_;
        std::vector<int64_t> kernelDataOffsets_;
        size_t noneZeroElements_;

        bool elementWiseChecks_(const Tensor& other);
        void elementWiseDouble_(const Tensor& other, const std::function<double(double, double)>& opFunc);
        void elementWiseInt_(const Tensor& other, const std::function<int64_t(int64_t, int64_t)>& opFunc);


        std::vector<std::size_t> indToSub_(const Shape& s, unsigned index) const;
        unsigned subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const;
        void populateSparsityMapTensor_();
        void setSubtensorsOrder_(Order order);

    public:
        std::vector<int64_t> getZeroPointsPerChannel() const;

        Tensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const mv::QuantizationParams& quantParams);
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

        // Returns true if the tensor was not sparse and sparsity was set, false otherwise
        bool setSparse();
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

        std::string subTensorInfo() const;

        inline bool isDoubleType() const {
            return getDType().isDoubleType();
        }

        std::vector<DataElement> getData();
        const std::vector<int64_t> getDataPacked();
        int getNumZeroPoints();
        const std::vector<int64_t> &getKernelDataOffsets();

        std::vector<double> getDoubleData();
        std::vector<int64_t> getIntData();
        void setDType(DType dType);
        DType getDType() const;
        void setOrder(Order order, bool updateSubtensors = false);
        Order getOrder() const;
        Shape getShape() const;
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

        DataElement& at(const std::vector<std::size_t>& sub);
        const DataElement& at(const std::vector<std::size_t>& sub) const;
        DataElement& at(std::size_t idx);
        const DataElement& at(std::size_t idx) const;
        DataElement& operator()(std::size_t idx);
        const DataElement& operator()(std::size_t idx) const;
        DataElement& operator()(const std::vector<std::size_t>& sub);
        const DataElement& operator()(const std::vector<std::size_t>& sub) const;

        inline bool hasSubTensors() const
        {
            bool flag = false;
            if (subTensors_.size() > 0)
                flag = true;
            return flag;
        }

        inline size_t numSubTensors() const
        {
            return subTensors_.size();
        }

        inline bool isQuantized() const
        {
            return hasAttr("quantParams");
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

        inline unsigned size() const
        {
            return shape_.totalSize();
        }

        inline unsigned dataPackedSize() const
        {
            return noneZeroElements_;
        }

        inline std::vector<std::size_t> indToSub(unsigned index) const
        {
            return indToSub_(shape_, index);
        }

        inline unsigned subToInd(const std::vector<std::size_t>& sub) const
        {
            return subToInd_(shape_, sub);
        }
        inline std::size_t getAddress() const
        {
            return get<std::size_t>("address");
        }

        std::shared_ptr<Tensor> getSparsityMap() const;
        std::shared_ptr<Tensor> getStorageElement() const;
        Tensor &getSubTensor(uint8_t cluster);

        Tensor& operator=(const Tensor& other);

        std::string toString() const override;
        virtual std::string getLogID() const override;

        std::vector<unsigned> computeNumericStrides() const;
        std::size_t computeTotalSize(unsigned int alignment = 16, bool base = false,
                                     bool fatherTensorAligned = false, bool graphOptimizer = false) const;
        std::size_t getClusterSize(unsigned int alignment = 16, bool base = false) const;
        void splitAcrossClusters(std::vector<Workload>, bool splitOverH, bool multicast);
        void shareAcrossClusters(std::vector<Workload>, unsigned int numClusters, bool clustering = true);
        void cleanSubtensors();
    };

}

#endif // TENSOR_HPP_
