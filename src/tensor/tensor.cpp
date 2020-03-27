#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"

std::map<std::string,mv::Tensor::MemoryLocation::Location> mv::Tensor::MemoryLocation::namingMap = mv::Tensor::MemoryLocation::createNamingMap();


mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order):
Element(name),
shape_(shape),
internalOrder_(Order(Order::getRowMajorID(shape.ndims()))),
blockSize_(shape[shape.ndims() - 1])
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    if(order.size() != shape.ndims())
        throw OrderError(*this, "Order and shape size are mismatching " + std::to_string(order.size()) + " vs " + std::to_string(shape.ndims()));
    set<Order>("order", order);
    if(dType == mv::DType("Default"))
        throw std::runtime_error("Tensors cannot be instantiated with default DType");
    set<DType>("dType", dType);
    set<bool>("populated", false);
    set<MemoryLocation>("Location",MemoryLocation::DEFAULT);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const mv::QuantizationParams &quantParams):
Tensor(name, shape, dType, order)
{
    set<mv::QuantizationParams>("quantParams", quantParams);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data) :
Tensor(name, shape, dType, order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data, const mv::QuantizationParams &quantParams) :
Tensor(name, shape, dType, order, quantParams)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<int64_t>& data) :
Tensor(name, shape, dType, order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<int64_t>& data, const mv::QuantizationParams &quantParams) :
Tensor(name, shape, dType, order, quantParams)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<mv::DataElement>& data) :
Tensor(name, shape, dType, order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<mv::DataElement>& data, const mv::QuantizationParams &quantParams):
Tensor(name, shape, dType, order, quantParams)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    populate(data, order);
}

mv::Tensor::Tensor(const Tensor &other) :
Element(other),
shape_(other.shape_),
internalOrder_(other.internalOrder_),
blockSize_(other.blockSize_),
sparsityMap_(other.sparsityMap_),
storageElement_(other.storageElement_),
subTensors_(),
kernelDataOffsets_(other.kernelDataOffsets_),
noneZeroElements_(other.noneZeroElements_)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)

    if (other.hasSubTensors())
    {
        for (std::size_t i = 0; i < other.numSubTensors(); i++)
        {
            subTensors_.push_back(std::make_shared<mv::Tensor>(*other.subTensors_[i]));
        }
    }


    if (other.isPopulated())
    {
        data_ = other.data_;
        blocks_ = other.blocks_;
    }
}

mv::Tensor::~Tensor()
{
}

std::vector<std::size_t> mv::Tensor::indToSub_(const Shape& s, unsigned index) const
{
    return internalOrder_.indToSub(s, index);
}

unsigned mv::Tensor::subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const
{

    if (hasAttr("master"))
    {
        Shape correctedShape(s);
        std::vector<std::size_t> correctedSub(sub);
        if (hasAttr("leftPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("leftPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
            {
                correctedShape[i] += padding[i];
                correctedSub[i] += padding[i];
            }
        }
        if (hasAttr("rightPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("rightPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
                correctedShape[i] += padding[i];

        }

        return internalOrder_.subToInd(correctedShape, correctedSub);

    }

    return internalOrder_.subToInd(s, sub);

}

void mv::Tensor::populate(const std::vector<double>& data)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (!isDoubleType())
        throw ArgumentError(*this, "data vector", "type double", "Unable to populate, data type is not double"
            "DType of tensor is " + getDType().toString() + " but populating with double data");

    if (data.size() != shape_.totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(shape_.totalSize()) + ")");

    data_ = std::make_shared<std::vector<DataElement>>(data.size(), DataElement(true));
    blocks_ = std::vector<std::vector<DataElement>::iterator>(shape_.totalSize() / blockSize_);

    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_->begin() + i * blockSize_;

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(shape_.ndims());
        for (std::size_t i = 0; i < data_->size(); ++i)
        {
            sub = getOrder().indToSub(shape_, i);
            data_->at(internalOrder_.subToInd(shape_, sub)) = data[i];
        }
    }
    else
        for (size_t i=0; i < data.size(); i++)
            data_->at(i) = data[i];


    set("populated", true);

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<mv::DataElement>& data)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (data.size() != shape_.totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(shape_.totalSize()) + ")");

    data_ = std::make_shared<std::vector<DataElement>>(data.size(), data[0].isDouble());
    blocks_ = std::vector<std::vector<DataElement>::iterator>(shape_.totalSize() / blockSize_);

    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_->begin() + i * blockSize_;

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(shape_.ndims());
        for (std::size_t i = 0; i < data_->size(); ++i)
        {
            sub = getOrder().indToSub(shape_, i);
            data_->at(internalOrder_.subToInd(shape_, sub)) = data[i];
        }
    }
    else
    {
        for (size_t i=0; i < data.size(); i++)
            data_->at(i) = data[i];
    }

    set("populated", true);

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<int64_t>& data)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (isDoubleType())
        throw ArgumentError(*this, "data vector", "type int", "Unable to populate, data type is not int"
            "DType of tensor is " + getDType().toString() + " but populating with int data");

    if (data.size() != shape_.totalSize())
    {
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(shape_.totalSize()) + ")");
    }

    data_ = std::make_shared<std::vector<DataElement>>(data.size(), false);
    blocks_ = std::vector<std::vector<DataElement>::iterator>(shape_.totalSize() / blockSize_);

    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_->begin() + i * blockSize_;

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(shape_.ndims());
        for (std::size_t i = 0; i < data_->size(); ++i)
        {
            sub = getOrder().indToSub(shape_, i);
            data_->at(internalOrder_.subToInd(shape_, sub)) = data[i];
        }
    }
    else
        for (size_t i=0; i < data.size(); i++)
            data_->at(i) = data[i];

    set("populated", true);

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<mv::DataElement>& data, Order order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    set<Order>("order", order);
    populate(data);
}

void mv::Tensor::populate(const std::vector<double>& data, Order order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    set<Order>("order", order);
    populate(data);
}

void mv::Tensor::populate(const std::vector<int64_t>& data, Order order)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    set<Order>("order", order);
    populate(data);
}

void mv::Tensor::unpopulate()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (!isPopulated())
        return;

    data_.reset();
    blocks_.clear();

    set<bool>("populated", false);

    //sparsity map tensor need to unpopulate as well
    if (isSparse())
    {
        sparsityMap_->unpopulate();
        kernelDataOffsets_.clear();
    }

}

// NOTE: why a reference? Otherwise return *this trick doesn't work
mv::Tensor& mv::Tensor::getSubTensor(uint8_t cluster)
{
    if (cluster < subTensors_.size())
        return *subTensors_[cluster];
    return *this;
}

std::shared_ptr<mv::Tensor> mv::Tensor::getSparsityMap() const
{
    if (!isSparse())
        throw ArgumentError(*this, "currentTensor", "SparsityMap" , " tensor not sparse, cannot get Sparsity Map");
    return sparsityMap_;
}

std::shared_ptr<mv::Tensor> mv::Tensor::getStorageElement() const
{
    if (!isSparse())
        throw ArgumentError(*this, "currentTensor", "storageElement" , " tensor not sparse, cannot get storage element");

    return storageElement_;
}

std::vector<int64_t> mv::Tensor::getZeroPointsPerChannel() const
{
    //default all zeropoints to zero
    std::vector<int64_t> zeroPoint(shape_[mv::KERNEL_OUTPUT_CHANNELS]);
    if (isQuantized())
    {
        auto quantParams = get<mv::QuantizationParams>("quantParams");
        for (size_t t=0; t < zeroPoint.size(); t++)
        {
            //Note: For now support of 1 channel zero point
            zeroPoint[t] = quantParams.getZeroPoint(0);
        }
    }
    return zeroPoint;
}

const mv::Order& mv::Tensor::getInternalOrder() const
{
    return internalOrder_;
}

// Auxiliary function to write sparsity map entry and reset all the variables
void writeMapEntry(std::vector<int64_t>& sparsityMapData, std::size_t& sparsityMapIdx, uint8_t& map, int& shift)
{
    sparsityMapData.at(sparsityMapIdx++) = map;
    map = 0;
    shift = 0;
}

void mv::Tensor::populateSparsityMapTensor_()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    auto shape = getShape();

    std::vector<int64_t> zeroPoint = getZeroPointsPerChannel();
    std::vector<int64_t> sparsityMapData(sparsityMap_->shape_.totalSize(), 0);
    std::vector<size_t> sub(shape.ndims());
    uint8_t map = 0;
    std::size_t sparsityMapIdx = 0;
    size_t n = 0;
    int shift = 0;
    int channelIndex = mv::KERNEL_OUTPUT_CHANNELS;
    noneZeroElements_ = 0;

    // Populating on a per channel basis
    for (size_t t = 0; t < shape.totalSize(); t++)
    {
        sub = getOrder().indToSub(shape, t);

        // Starting a new channel
        if (sub[channelIndex] != n)
        {
            // This is needed in the case when tensor dimensions are not multiple of 8
            // This should never happen because weights sets are padded to have alignment to 16
            if (shift != 0)
                writeMapEntry(sparsityMapData, sparsityMapIdx, map, shift);

            // Updating current channel index
            n = sub[channelIndex];

            // Again, this should never happen, same reasoning as above
            if (sparsityMapIdx % 16 != 0)
            {
                auto padding = 16 - (sparsityMapIdx % 16);
                sparsityMapIdx += padding;
            }
        }

        // Updating current entry: 1 UInt8 contains 8 bits, so can cover for 8 elements
        // NOTE: NoneZero elements can't be counted here
        // Because we need alignment to 16, which can be obtained only in getPackedData.
        if (static_cast<int64_t>(data_->at(internalOrder_.subToInd(shape, sub))) != zeroPoint[sub[channelIndex]])
            map ^= (1 << shift);

        // Finished one entry, writing it and resetting entry and shift variables
        if (++shift == 8)
            writeMapEntry(sparsityMapData, sparsityMapIdx, map, shift);
    }

    // Following the above reasoning, this should never happen
    if (shift != 0)
        writeMapEntry(sparsityMapData, sparsityMapIdx, map, shift);

    sparsityMap_->populate(sparsityMapData);
}

void mv::Tensor::setAddress(int64_t address)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    set<std::size_t>("address", address);
    if (isSparse())
    {
        auto tensorSize = getClusterSize();

        // Order assumed for unpopulated: Tensor - Storage Element - Sparsity Map
        // Order assumed for populated: Tensor (packed data) - Sparsity Map
        auto sparsitySize = sparsityMap_->getClusterSize();

        if(!isPopulated())
        {
            auto storageElementSize = storageElement_->getClusterSize();
            storageElement_->setAddress(address + (tensorSize - storageElementSize - sparsitySize));
        }
        sparsityMap_->setAddress(address + (tensorSize - sparsitySize));
    }
    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setAddress(address);
}

bool mv::Tensor::setSparse()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    mv::Order order =  getOrder();
    if (!order.isZMajor())
        throw ArgumentError(*this, "Order", order.toString() , " Sparsity requires ZMajor layout (NHWC)");

    if (order.size() < 3)
        throw ArgumentError(*this, "Order", order.toString() , " Sparsity requires order of size >= 3");

    if(hasAttr("sparse") && get<bool>("sparse") == true)
        return false;

    set<bool>("sparse", true);

    auto shape = getShape();
    size_t N = shape[shape.ndims()-1];

    //Sparsity map
    //we choose layout as internal layout, no need to reorder
    mv::Shape mapShape({shape[0], shape[1], static_cast<std::size_t>(std::ceil(shape[2] / 8.0)), N});
    if (isPopulated())
    {
        //pad the shape
        auto paddedDim = mv::round_up(static_cast<std::size_t>(std::ceil(shape[0] / 8.0)), 16);
        mapShape = {paddedDim, 1, 1, N};
    }
    //std::cout << "Total bytes of sparsity map for " << getName() << " " << mapShape.totalSize() * mv::DType("UInt8").getSizeInBits() / 8 << std::endl;
    sparsityMap_ = std::make_shared<Tensor>(createSparsityMapName(getName()), mapShape, mv::DType("UInt8"), Order("NHWC"));
    noneZeroElements_ = 0;

    //populate sparsity map
    if (isPopulated())
    {
        populateSparsityMapTensor_();

        // NOTE: Necessary to initialize correctly noneZeroElements_
        getDataPacked();
    }
    else
    {
        mv::Shape storageElementShape({shape[0], shape[1], 1, N});
        //std::cout << "Total bytes of storage element for " << getName() << " " << storageElementShape.totalSize() * mv::DType("Int32").getSizeInBits() / 8 << std::endl;
        storageElement_  = std::make_shared<Tensor>(createStorageElementName(getName()), storageElementShape, mv::DType("Int32"), order);
    }

    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setSparse();

    return true;
}

void mv::Tensor::bindData(Tensor& other, const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t> &rightPadding)
{
    if (leftPadding.size() != other.shape_.ndims() && !leftPadding.empty())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Invalid dimension of the left padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(other.shape_.ndims()));

    if (rightPadding.size() != other.shape_.ndims() && !rightPadding.empty())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Invalid dimension of the right padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(shape_.ndims()));

    if (other.isPopulated())
    {
        data_ = other.data_;
        set<bool>("populated", true);
    }

    if (!leftPadding.empty())
        set<std::vector<std::size_t>>("leftPadding", leftPadding);
    if (!rightPadding.empty())
        set<std::vector<std::size_t>>("rightPadding", rightPadding);

    Shape newShape(other.shape_);

    for (std::size_t i = 0; i < leftPadding.size(); ++i)
        newShape[i] -= leftPadding[i] + rightPadding[i];

    set<Shape>("shape", newShape);
    set<Order>("order", other.getOrder());
    set<DType>("dType", other.getDType());
    set<std::string>("master", other.getName());
    if (!other.hasAttr("slave"))
        other.set<std::vector<std::string>>("slave", { getName() });
    else
        other.get<std::vector<std::string>>("slave").push_back(getName());

    if (other.hasAttr("quantParams"))
        set<mv::QuantizationParams>("quantParams", other.get<mv::QuantizationParams>("quantParams"));
}

void mv::Tensor::setOrder(Order order, bool updateSubtensors)
{
    if(order != getOrder())
    {
        set<Order>("order", order);

        //If the data order changes, the packed data is invalid
        if (updateSubtensors)
            setSubtensorsOrder_(order);
    }
}

void mv::Tensor::setSubtensorsOrder_(Order order)
{
    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setOrder(order);
}

void mv::Tensor::setDType(DType dtype)
{

    set<DType>("dType", dtype);
    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setDType(dtype);
    return;

}

void mv::Tensor::setShape(const Shape& shape)
{
    if(isPopulated())
    {
        if(shape.totalSize() != getShape().totalSize())
            throw ArgumentError(*this, "CurrentTensor", "shape", "Unable to change shape of a populated tensor");
    }
    shape_ = shape;
    return;
}


void mv::Tensor::broadcast(const Shape& shape)
{

    if (!isPopulated())
        throw ValueError(*this, "Broadcastring of an unpopulated tensor is undefined");

    if (hasAttr("master") || hasAttr("slave"))
        throw ValueError(*this, "Unable to broadcast a bound tensor");

    Shape s1 = shape_, s2 = shape;

    if (s1.ndims() == 0 || s2.ndims() == 0)
        throw ValueError(*this, "Unable to perfom element-wise operation using 0-dimensional tensor");

    if (s1 != s2)
    {

        // Will throw on error
        Shape sO = Shape::broadcast(s1, s2);
        std::vector<DataElement> dataBuf = std::vector<DataElement>(sO.totalSize(), DataElement(isDoubleType()));

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        if (sO.totalSize() != s1.totalSize())
            data_->resize(dataBuf.size(), isDoubleType());

        for (unsigned i = 0; i < dataBuf.size(); ++i)
        {

            std::vector<std::size_t> sub = indToSub_(sO, i);

            for (unsigned j = 0; j < sub.size(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub[j] = 0;
            }

            dataBuf[i] = at(subToInd_(s1, sub));

        }

        set<Shape>("shape", sO);
        shape_ = sO;
        for (unsigned i = 0; i < dataBuf.size(); ++i)
            data_->at(i) = dataBuf[i];
    }

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
std::vector<double> mv::Tensor::getDoubleData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (!isDoubleType())
        throw ValueError(*this, "Attempt of reading double data from an int type tensor");

    std::vector<double> orderedData(shape_.totalSize());

    std::vector<std::size_t> sub(shape_.ndims());
    for (std::size_t i = 0; i < data_->size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(shape_, i);
            orderedData[getOrder().subToInd(shape_, sub)] = data_->at(i);
        }
        else
            orderedData[i] = data_->at(i);
    }

    return orderedData;
}

std::vector<mv::DataElement> mv::Tensor::getData()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    std::vector<DataElement> orderedData(shape_.totalSize(), DataElement(isDoubleType()));

    std::vector<std::size_t> sub(shape_.ndims());
    for (std::size_t i = 0; i < data_->size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(shape_, i);
            orderedData[getOrder().subToInd(shape_, sub)] = data_->at(i);
        }
        else
            orderedData[i] = data_->at(i);
    }

    return orderedData;
}

const std::vector<int64_t>& mv::Tensor::getKernelDataOffsets()
{
    if(kernelDataOffsets_.empty())
        getDataPacked();
    return kernelDataOffsets_;
}

// NOTE: For now this can handle only integer tensors
// This is because we have to make quantParams (currently int64) dataElements as well.

// NOTE2: This method works properly only on tensors of the form (W, 1, 1, N)
// The order has to be an order in which the W comes after the N, examples:
// NHWC, NHCW
const std::vector<int64_t> mv::Tensor::getDataPacked()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    // TODO: Make it work on doubles - Or maybe not considering that FP16 is integer type
    if(isDoubleType())
        throw ValueError(*this, "Attempt of getting data packed from populated double tensor");

    if(!getOrder().isZMajor())
        throw ValueError(*this, "To use getDataPacked order has to be NHWC");

    std::vector<int64_t> orderedDataPacked;

    // NOTE: noneZeroElements_ has a misleading name. We pad the packed data
    // and that actually is memorized in graphfile and transferred. So it's
    // counter must be updated for both elements != ZeroPoint and for each
    // ZeroPoint of padding that we put in orderedDataPacked
    noneZeroElements_ = 0;

    auto shape = getShape();
    std::vector<std::size_t> sub(shape.ndims());
    std::vector<int64_t> zeroPoint = getZeroPointsPerChannel();

    int64_t datai;
    size_t outputChannels = shape[mv::KERNEL_OUTPUT_CHANNELS];
    size_t outputChannelSize = shape.totalSize() / outputChannels;
    kernelDataOffsets_.resize(outputChannels);
    size_t offset = 0;

    // Filling the data on a per channel basis
    for (std::size_t k = 0; k < outputChannels; ++k)
    {
        kernelDataOffsets_[k] = offset;
        size_t prevNumOfElements = orderedDataPacked.size();

        for (std::size_t i = 0; i < outputChannelSize; i++)
        {
            sub = getOrder().indToSub(shape, k*outputChannelSize + i);
            datai = static_cast<int64_t>(data_->at(internalOrder_.subToInd(shape, sub)));
            //skip zero values if sparse
            if (!isSparse() || datai != zeroPoint[sub[mv::KERNEL_OUTPUT_CHANNELS]])
            {
                orderedDataPacked.push_back(datai);
                noneZeroElements_++;
            }
        }

        // Add padding if needed - Needs to be done only in the sparse case
        // As weights sets are aligned to 16
        if (isSparse())
        {
            auto size = orderedDataPacked.size() * std::ceil(getDType().getSizeInBits()/8.0);
            auto padsize = mv::round_up(size, 16) - size;
            int64_t zeroPointVal = zeroPoint[sub[mv::KERNEL_OUTPUT_CHANNELS]];
            for (std::size_t j = 0; j < padsize; ++j)
            {
                orderedDataPacked.push_back(zeroPointVal);
                noneZeroElements_++;
            }
        }

        size_t numberOfElementsInKernel = orderedDataPacked.size() - prevNumOfElements; //include padding
        offset += numberOfElementsInKernel * std::ceil(getDType().getSizeInBits()/8.0);
    }

    return orderedDataPacked;
}

int mv::Tensor::getNumZeroPoints()
{
    int numZeroPoints = 0; 

    auto shape = getShape();
    std::vector<std::size_t> sub(shape.ndims());
    std::vector<int64_t> zeroPoint = getZeroPointsPerChannel();

    int64_t datai;
    size_t outputChannels = shape[mv::KERNEL_OUTPUT_CHANNELS];
    size_t outputChannelSize = shape.totalSize() / outputChannels;

    // Filling the data on a per channel basis
    for (std::size_t k = 0; k < outputChannels; ++k)
    {
        for (std::size_t i = 0; i < outputChannelSize; i++)
        {
            sub = getOrder().indToSub(shape, k*outputChannelSize + i);
            datai = static_cast<int64_t>(data_->at(internalOrder_.subToInd(shape, sub)));
            if ( datai == zeroPoint[sub[mv::KERNEL_OUTPUT_CHANNELS]] )
                numZeroPoints++;
        }
    }

    return numZeroPoints;
}

std::vector<int64_t> mv::Tensor::getIntData()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BULD)
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (isDoubleType())
        throw ValueError(*this, "Attempt of reading int data from an double type tensor");

    std::vector<int64_t> orderedData(shape_.totalSize());

    std::vector<std::size_t> sub(shape_.ndims());
    for (std::size_t i = 0; i < data_->size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(shape_, i);
            orderedData[getOrder().subToInd(shape_, sub)] = data_->at(i);
        }
        else
        {
            orderedData[i] = data_->at(i);
        }

    }
    return orderedData;
}

mv::DType mv::Tensor::getDType() const
{
    return get<DType>("dType");
}

mv::Order mv::Tensor::getOrder() const
{
    return get<Order>("order");
}

std::string mv::Tensor::toString() const
{
    return getLogID() + Element::attrsToString_();
}

std::string mv::Tensor::subTensorInfo() const
{
    std::string toReturn;
    if (hasSubTensors())
    {
        for (int i = 0; i < numSubTensors(); i++)
        {
            toReturn += subTensors_[i]->getShape().toString();
        }
    }

    return toReturn;
}

bool mv::Tensor::elementWiseChecks_(const Tensor& other)
{
    if (!isPopulated() || !other.isPopulated())
        throw ValueError(*this, "Unable to perfom element-wise operation using unpopulated tensor");

    Shape s1 = shape_, s2 = other.shape_;

    if (s1.ndims() == 0)
        throw ArgumentError(*this, "tensor:Shape:ndims", std::to_string(s1.ndims()),
             "0-dimensional tensor is illegal for element-wise operation");

    if (s2.ndims() == 0)
        throw ArgumentError(*this, "input tensor:Shape:ndims", std::to_string(s2.ndims()),
             "0-dimensional tensor is illegal for element-wise operation");

    if (s2.ndims() > s1.ndims())
        throw ArgumentError(*this, "input tensor:Shape:ndims", s2.toString(),
            "Currently unsupported in element wise in combination with " + s1.toString());

    for (std::size_t i = 1; i <= s2.ndims(); ++i)
        if (s1[-i] != s2[-i])
            throw ArgumentError(*this, "input tensor:Shape", s2.toString(),
                "Currently unsupported in combination with " + s1.toString());

    return (s1 == s2);
}

void mv::Tensor::elementWiseInt_(const Tensor& other, const std::function<int64_t(int64_t, int64_t)>& opFunc)
{
    if (elementWiseChecks_(other))
        std::transform(data_->begin(), data_->end(), other.data_->begin(), data_->begin(), opFunc);
    else
    {

        std::size_t firstIdx = 0;
        while (firstIdx < blocks_.size())
        {
            for (std::size_t secondIdx = 0; secondIdx < other.blocks_.size(); ++secondIdx)
            {
                std::transform(blocks_[firstIdx], blocks_[firstIdx] + blockSize_, other.blocks_[secondIdx], blocks_[firstIdx], opFunc);
                ++firstIdx;
            }
        }

    }

}
void mv::Tensor::elementWiseDouble_(const Tensor& other, const std::function<double(double, double)>& opFunc)
{
    if (elementWiseChecks_(other))
        std::transform(data_->begin(), data_->end(), other.data_->begin(), data_->begin(), opFunc);
    else
    {

        std::size_t firstIdx = 0;
        while (firstIdx < blocks_.size())
        {
            for (std::size_t secondIdx = 0; secondIdx < other.blocks_.size(); ++secondIdx)
            {
                std::transform(blocks_[firstIdx], blocks_[firstIdx] + blockSize_, other.blocks_[secondIdx], blocks_[firstIdx], opFunc);
                ++firstIdx;
            }
        }

    }

}

void mv::Tensor::add(const Tensor& other)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (isDoubleType())
        elementWiseDouble_(other, std::plus<double>());
    else
        elementWiseInt_(other, std::plus<int64_t>());
}

void mv::Tensor::add(double val)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar addition operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_->size(); ++i)
        data_->at(i) += val;
}

void mv::Tensor::subtract(const Tensor& other)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (isDoubleType())
        elementWiseDouble_(other, std::minus<double>());
    else
        elementWiseInt_(other, std::minus<int64_t>());

}

void mv::Tensor::subtract(double val)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar subtraction operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        data_->at(i) -= val;
}

void mv::Tensor::multiply(const Tensor& other)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (isDoubleType())
        elementWiseDouble_(other, std::multiplies<double>());
    else
        elementWiseInt_(other, std::multiplies<int64_t>());

}

void mv::Tensor::multiply(double val)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar multiplication operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_->size(); ++i)
        data_->at(i) *= val;

}

void mv::Tensor::divide(double val)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar division operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_->size(); ++i)
        data_->at(i) /= val;

}

void mv::Tensor::divide(const Tensor& other)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (isDoubleType())
        elementWiseDouble_(other, std::divides<double>());
    else
        elementWiseInt_(other, std::divides<int64_t>());

}

void mv::Tensor::sqrt()
{
    MV_PROFILED_FUNCTION(MV_PROFILE_MATH)
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar square root operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_->size(); ++i)
        if (isDoubleType())
            data_->at(i) = std::sqrt(static_cast<double>(data_->at(i)));
        else
            data_->at(i) = std::sqrt(static_cast<int64_t>(data_->at(i)));

}

mv::DataElement& mv::Tensor::at(const std::vector<std::size_t>& sub)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");
    return data_->at(subToInd(sub));
}

const mv::DataElement& mv::Tensor::at(const std::vector<std::size_t>& sub) const
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    return data_->at(subToInd(sub));
}

mv::DataElement& mv::Tensor::at(std::size_t idx)
{
    return const_cast<DataElement&>(static_cast<const Tensor*>(this)->at(idx));
}

const mv::DataElement& mv::Tensor::at(std::size_t idx) const
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    if (idx > data_->size())
        throw IndexError(*this, idx, "Exceeds the total lenght of data vector");

    if (hasAttr("master"))
    {
        std::vector<std::size_t> sub = indToSub(idx);
        return data_->at(subToInd(sub));
    }

    if (getOrder() == internalOrder_)
        return data_->at(idx);

    auto sub = getOrder().indToSub(shape_, idx);
    return data_->at(internalOrder_.subToInd(shape_, sub));
}

mv::DataElement& mv::Tensor::operator()(std::size_t idx)
{
    return at(idx);
}

const mv::DataElement& mv::Tensor::operator()(std::size_t idx) const
{
    return at(idx);
}

mv::DataElement& mv::Tensor::operator()(const std::vector<std::size_t>& sub)
{
    return at(sub);
}

mv::Tensor& mv::Tensor::operator=(const Tensor& other)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    Element::operator=(other);
    shape_ = other.shape_;
    internalOrder_ = other.internalOrder_;
    blockSize_ = other.blockSize_;
    sparsityMap_ = other.sparsityMap_;
    storageElement_ = other.storageElement_;
    subTensors_ = other.subTensors_;
    kernelDataOffsets_ = other.kernelDataOffsets_;

    if (other.isPopulated())
    {
        data_ = other.data_;
        blocks_ = other.blocks_;
    }

    return *this;

}

const mv::DataElement& mv::Tensor::operator()(const std::vector<std::size_t>& sub) const
{
    return at(sub);
}

std::string mv::Tensor::getLogID() const
{
    return "Tensor:" + getName();
}

std::vector<unsigned> mv::Tensor::computeNumericStrides() const
{
    return getOrder().computeByteStrides(shape_, getDType().getSizeInBits() / 8);
}


//isBase == true for implicit operations.
std::size_t mv::Tensor::getClusterSize(unsigned int alignment, bool isBase) const
{
    std::size_t res;
    if (!isBroadcasted())
    {
        res = 0;
        bool isTensorAligned = hasAttr("alignment") ? get<bool>("alignment") : false;

        for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        {
            auto size = subTensors_[tIdx]->computeTotalSize(alignment, isBase, isTensorAligned);
            if (size > res)
                res = size;
        }
    }
    else
    {
        res = computeTotalSize(alignment, isBase);
    }

    return res;
}

std::size_t mv::Tensor::computeTotalSize(unsigned int alignment, bool isBase, bool fatherTensorAligned
                                         , bool graphOptimizer) const
{
    std::size_t res;

    auto shape = getShape();

    //use shape of master
    if (!isBase && hasAttr("master"))
    {
        if (hasAttr("leftPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("leftPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
            {
                shape[i] += padding[i];
            }
        }
        if (hasAttr("rightPadding"))
        {
            auto padding = get<std::vector<std::size_t>>("rightPadding");
            for (std::size_t i = 0; i < padding.size(); ++i)
                shape[i] += padding[i];

        }
    }
    //NOTE :: Graph Optimizer MemorySize does not know about the alignment of the input/output Channels
    bool isTensorAligned = hasAttr("alignment") ? get<bool>("alignment") : false;
    size_t totalSize = shape.totalSize();
    //TODO update that to proper alignment, if this needs to align to 32 (splitOverK, each cluster has the whole tensor size, but need it aligned to 16*numclusters)
    if (isTensorAligned || fatherTensorAligned)
    {
        auto pad = alignment;
        auto outputChannels = shape[mv::IO_CHANNEL_DIMENSION];
        if (outputChannels % pad != 0)
        {
            auto paddedOutputChannels = mv::round_up(outputChannels, pad);
            totalSize = totalSize / outputChannels * paddedOutputChannels;
        }
    }
    else if (graphOptimizer)
    {
        auto pad = alignment;
        auto outputChannels = shape[mv::IO_CHANNEL_DIMENSION];
        if (outputChannels % pad != 0)
        {
            auto paddedOutputChannels = mv::round_up(outputChannels, pad);
            totalSize = totalSize / outputChannels * paddedOutputChannels;
        }
    }
    if (isSparse())
    {
        if (isPopulated())
        {
            res = noneZeroElements_ * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
            res += getSparsityMap()->computeTotalSize();
        }
        else
        {
            //TODO what happens in this case
            res = shape.totalSize() * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
            res += getSparsityMap()->computeTotalSize();
            res += getStorageElement()->computeTotalSize();
        }
    }
    else
    {
        res = totalSize * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
    }
    //Round up to align to (alignment) 16 bytes
    res = mv::round_up(res, alignment);
    return res;
}

void mv::Tensor::shareAcrossClusters(std::vector<mv::Workload> workloads, unsigned int numClusters, bool clustering)
{
    if (isPopulated())
    {
        //NOTE Shape of Populated will be aligned already, so I am using the shape not the workload
        auto shape = getShape();
        for (auto wlItr = workloads.begin(); wlItr != workloads.end(); wlItr++)
        {
            size_t idx = wlItr - workloads.begin();

            if (clustering)
            {
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                    shape_, getDType(), getOrder()));
            }
            if (hasAttr("quantizationParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantizationParams", get<mv::QuantizationParams>("quantizationParams"));
            if (isSparse())
                subTensors_[idx]->setSparse();
        }
        set<bool>("broadcasted", clustering == true && (numClusters > 1));
    }
    else
    {
        auto shape = getShape();
        for (auto wlItr = workloads.begin(); wlItr != workloads.end(); wlItr++)
        {
            size_t idx = wlItr - workloads.begin();
            auto width = wlItr->MaxX - wlItr->MinX;
            auto height = wlItr->MaxY - wlItr->MinY;
            auto channels = wlItr->MaxZ - wlItr->MinZ;
            if (clustering)
            {
                mv::Shape newShape = {width, height, channels, 1};
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx), newShape, getDType(), getOrder()));
           }

            if (hasAttr("quantizationParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantizationParams", get<mv::QuantizationParams>("quantizationParams"));
            if (isSparse())
                subTensors_[idx]->setSparse();
        }
        set<bool>("broadcasted", (clustering && (numClusters > 1)));
    }
}

void mv::Tensor::splitAcrossClusters(std::vector<mv::Workload> workloads, bool splitOverH, bool multicast)
{
    if (isPopulated())
    {
        auto shape = getShape();
        for (auto wlItr = workloads.begin(); wlItr != workloads.end(); wlItr++)
        {
            size_t idx = wlItr - workloads.begin();
            auto width = wlItr->MaxX - wlItr->MinX + 1;
            auto height = wlItr->MaxY - wlItr->MinY + 1;
            if (splitOverH)
            {
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                    shape_, getDType(), getOrder()));
            }
            else
            {
                mv::Shape newShape = { shape[0], shape[1] , static_cast<size_t>(width), static_cast<size_t>(height)};
                auto order = getOrder();

                // NOTE: Physically copying the data to subtensors is needed only when the tensor is sparse
                // since we need to sparsify the subtensors to get the kernel data offsets.
                if (isSparse())
                {
                    std::vector<mv::DataElement> splittedData(newShape.totalSize(), mv::DataElement(this->isDoubleType()));
                    size_t nOffset = static_cast<size_t>(wlItr->MinY);
                    size_t cOffset = static_cast<size_t>(wlItr->MinX);
                    for (size_t n = 0; n < newShape[3]; n++)
                        for (size_t c = 0; c < newShape[2]; c++)
                            for (size_t h = 0; h < newShape[1]; h++)
                                for (size_t w = 0; w < newShape[0]; w++)
                                    splittedData[order.subToInd(newShape, {w, h, c, n})] = this->at({w , h, c+cOffset, n+nOffset});
                    subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                        newShape, getDType(), order, splittedData));
                }
                if (isSplitOverK()) 
                {
                    std::vector<mv::DataElement> splittedData(newShape.totalSize(), mv::DataElement(this->isDoubleType()));
                    size_t nOffset = static_cast<size_t>(wlItr->MinY);
                    size_t cOffset = static_cast<size_t>(wlItr->MinX);
                    for (size_t n = 0; n < newShape[3]; n++)
                        for (size_t c = 0; c < newShape[2]; c++)
                            for (size_t h = 0; h < newShape[1]; h++)
                                for (size_t w = 0; w < newShape[0]; w++)
                                    splittedData[order.subToInd(newShape, {w, h, c, n})] = this->at({w , h, c+cOffset, n+nOffset});
                    subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                        newShape, getDType(), order, splittedData));
                }
                else
                {
                    subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                        newShape, getDType(), order));
                }
            }
            std::vector<std::size_t> offset = {0 , 0,
                static_cast<size_t>(wlItr->MinX), static_cast<size_t>(wlItr->MinY)};
            subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);

            if (hasAttr("quantParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantParams", get<mv::QuantizationParams>("quantParams"));
            if (isSparse())
                subTensors_[idx]->setSparse();
        }
        set<bool>("broadcasted", (splitOverH == true));
    }
    else
    {
        auto shape = getShape();
        for (auto wlItr = workloads.begin(); wlItr != workloads.end(); wlItr++)
        {
            size_t idx = wlItr - workloads.begin();
            auto width = wlItr->MaxX - wlItr->MinX + 1;
            auto height = wlItr->MaxY - wlItr->MinY + 1;
            if (splitOverH)
            {
                mv::Shape newShape = { static_cast<size_t>(width), static_cast<size_t>(height) ,shape[2], shape[3]};
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx), newShape, getDType(), getOrder()));
                std::vector<std::size_t> offset = {static_cast<size_t>(wlItr->MinX), static_cast<size_t>(wlItr->MinY), 0 , 0};
                subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);
            }
            else
            {
                mv::Shape newShape = { shape[0], shape[1] , static_cast<size_t>(width), static_cast<size_t>(height)};
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx), newShape, getDType(), getOrder()));
                std::vector<std::size_t> offset =  {0 , 0, static_cast<size_t>(wlItr->MinX), static_cast<size_t>(wlItr->MinY)};
                subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);
            }

            if (hasAttr("quantParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantParams", get<mv::QuantizationParams>("quantParams"));
            if (isSparse())
                subTensors_[idx]->setSparse();

            if (!splitOverH || multicast)
            {
                std::vector<std::size_t> leftPadding = subTensors_[idx]->get<std::vector<std::size_t>>("offset");
                std::vector<std::size_t> rightPadding(shape.ndims());
                for (size_t i = 0; i < rightPadding.size(); i++)
                    rightPadding[i] = shape[i] - leftPadding[i] - subTensors_[idx]->shape_[i];
                subTensors_[idx]->bindData(*this, leftPadding, rightPadding);
            }
        }

        set<bool>("broadcasted", (!splitOverH || multicast));
    }
}

void mv::Tensor::cleanSubtensors()
{
        subTensors_ = {};
}

mv::Shape mv::Tensor::getShape() const
{
    return shape_;
}

int mv::Tensor::computeAppropriatePadding() const
{
    int pad;
    if (getDType() == mv::DType("Float16"))
        pad = 8;
    else if (getDType() == mv::DType("UInt8"))
        pad = 16;
    return pad;
}
