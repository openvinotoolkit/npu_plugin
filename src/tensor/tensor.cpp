#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order):
Element(name),
blockSize_(shape[-1]),
blocks_(shape.totalSize() / blockSize_),
shape_(shape),
internalOrder_(Order(Order::getRowMajorID(shape.ndims())))
{

    log(Logger::MessageType::Debug, "Initialized");
    if(order.size() != shape.ndims())
        throw OrderError(*this, "Order and shape size are mismatching " + std::to_string(order.size()) + " vs " + std::to_string(shape.ndims()));
    set<Shape>("shape", shape_);
    set<Order>("order", order);
    set<DType>("dType", dType);
    set<bool>("populated", false);
    //set<bool>("broadcasted", true);

    data_ = std::vector<DataElement>(shape.totalSize(), DataElement(isDoubleType()));
    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_.begin() + i * blockSize_;
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const mv::QuantizationParams &quantParams):
Element(name),
blockSize_(shape[-1]),
blocks_(shape.totalSize() / blockSize_),
shape_(shape),
internalOrder_(Order(Order::getRowMajorID(shape.ndims())))
{

    log(Logger::MessageType::Debug, "Initialized");
    if(order.size() != shape.ndims())
        throw OrderError(*this, "Order and shape size are mismatching " + std::to_string(order.size()) + " vs " + std::to_string(shape.ndims()));
    set<Shape>("shape", shape_);
    set<Order>("order", order);
    set<DType>("dType", dType);
    set<mv::QuantizationParams>("quantParams", quantParams);
    set<bool>("populated", false);
    //set<bool>("broadcasted", true);

    data_ = std::vector<DataElement>(shape.totalSize(), DataElement(isDoubleType()));
    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_.begin() + i * blockSize_;
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const mv::QuantizationParams &quantParams, bool flag):
Element(name),
blockSize_(shape[-1]),
blocks_(shape.totalSize() / blockSize_),
shape_(shape),
internalOrder_(Order(Order::getRowMajorID(shape.ndims())))
{

    log(Logger::MessageType::Debug, "Initialized");
    if(order.size() != shape.ndims())
        throw OrderError(*this, "Order and shape size are mismatching " + std::to_string(order.size()) + " vs " + std::to_string(shape.ndims()));
    set<Shape>("shape", shape_);
    set<Order>("order", order);
    set<DType>("dType", dType);
    set<mv::QuantizationParams>("quantParams", quantParams);
    set<bool>("populated", flag);

    data_ = std::vector<DataElement>(shape.totalSize(), DataElement(isDoubleType()));
    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_.begin() + i * blockSize_;
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data) :
Tensor(name, shape, dType, order)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data, const mv::QuantizationParams &quantParams) :
Tensor(name, shape, dType, order, quantParams, true)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<int64_t>& data) :
Tensor(name, shape, dType, order)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<int64_t>& data, const mv::QuantizationParams &quantParams) :
Tensor(name, shape, dType, order, quantParams, true)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<mv::DataElement>& data) :
Tensor(name, shape, dType, order)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<mv::DataElement>& data, const mv::QuantizationParams &quantParams):
Tensor(name, shape, dType, order, quantParams, true)
{
    populate(data, order);
}

mv::Tensor::Tensor(const Tensor &other) :
Element(other),
data_(other.data_.size(), other.isDoubleType()),
blockSize_(other.blockSize_),
blocks_(other.blocks_.size()),
shape_(other.shape_),
internalOrder_(Order(Order::getRowMajorID(other.shape_.ndims())))
{

    log(Logger::MessageType::Debug, "Copied");
    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_.begin() + i * blockSize_;

    if (isPopulated())
        data_ = other.data_;
    if (other.isSparse())
    {
        sparsityMap_ = std::make_shared<Tensor>(*other.sparsityMap_);
        storageElement_ = std::make_shared<Tensor>(*other.storageElement_);
        noneZeroElements_ = other.noneZeroElements_;
    }
    if (other.subTensors_.size() > 0)
    {
        for (size_t tIdx = 0; tIdx < other.subTensors_.size(); tIdx++)
        {
            subTensors_.push_back(std::make_shared<Tensor>(*other.subTensors_[tIdx]));
        }
    }
}

mv::Tensor::~Tensor()
{
    log(Logger::MessageType::Debug, "Deleted");
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
    if (!isDoubleType())
        throw ArgumentError(*this, "data vector", "type double", "Unable to populate, data type is not double"
            "DType of tensor is " + getDType().toString() + " but populating with double data");

    if (data.size() != getShape().totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(getShape().totalSize()) + ")");

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(getShape().ndims());
        for (std::size_t i = 0; i < data_.size(); ++i)
        {
            sub = getOrder().indToSub(getShape(), i);
            data_[internalOrder_.subToInd(getShape(), sub)] = data[i];
        }
    }
    else
        for (size_t i=0; i < data.size(); i++)
            data_[i] = data[i];


    set("populated", true);
    log(Logger::MessageType::Debug, "Populated");

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<mv::DataElement>& data)
{
    if (data.size() != getShape().totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(getShape().totalSize()) + ")");

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(getShape().ndims());
        for (std::size_t i = 0; i < data_.size(); ++i)
        {
            sub = getOrder().indToSub(getShape(), i);
            data_[internalOrder_.subToInd(getShape(), sub)] = data[i];
        }
    }
    else
        for (size_t i=0; i < data.size(); i++)
            data_[i] = data[i];


    set("populated", true);
    log(Logger::MessageType::Debug, "Populated");

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<int64_t>& data)
{
    if (isDoubleType())
        throw ArgumentError(*this, "data vector", "type int", "Unable to populate, data type is not int"
            "DType of tensor is " + getDType().toString() + " but populating with int data");

    if (data.size() != getShape().totalSize())
        throw ArgumentError(*this, "data vector", std::to_string(data.size()), "Unable to populate, data vector size"
            "does not match total size the tensor (" + std::to_string(getShape().totalSize()) + ")");

    if (getOrder() != internalOrder_)
    {
        std::vector<std::size_t> sub(getShape().ndims());
        for (std::size_t i = 0; i < data_.size(); ++i)
        {
            sub = getOrder().indToSub(getShape(), i);
            data_[internalOrder_.subToInd(getShape(), sub)] = data[i];
        }
    }
    else
        for (size_t i=0; i < data.size(); i++)
            data_[i] = data[i];

    set("populated", true);
    log(Logger::MessageType::Debug, "Populated");

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
}

void mv::Tensor::populate(const std::vector<mv::DataElement>& data, Order order)
{
    set<Order>("order", order);
    populate(data);
}

void mv::Tensor::populate(const std::vector<double>& data, Order order)
{
    set<Order>("order", order);
    populate(data);
}

void mv::Tensor::populate(const std::vector<int64_t>& data, Order order)
{
    set<Order>("order", order);
    populate(data);
}

int mv::Tensor::computeMemoryRequirement() const
{
    //TODO I think this needs to be updated to use clusterSize
    return (shape_.totalSize() * get<DType>("dType").getSizeInBits() / 8);
}

void mv::Tensor::unpopulate()
{
    if (!isPopulated())
        return;

    data_.clear();

    set<bool>("populated", false);

    //sparsity map tensor need to unpopulate as well
    if (isSparse())
        sparsityMap_->unpopulate();

    log(Logger::MessageType::Debug, "Unpopulated");
}

const mv::Tensor& mv::Tensor::getSubTensor(uint8_t cluster)
{
    if (cluster < subTensors_.size() && !isBroadcasted())
        return *subTensors_[cluster];
    return *this;
}

const mv::Tensor& mv::Tensor::broadcastSubtensor(uint8_t cluster)
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

// NOTE: Read NOTE in header fileq
std::vector<int64_t> mv::Tensor::getZeroPointsPerChannel()
{
    //default all zeropoints to zero
    std::vector<int64_t> zeroPoint(getShape()[mv::KERNEL_OUTPUT_CHANNELS]);
    if (isQuantized())
    {
        auto quantParams = get<mv::QuantizationParams>("quantParams");
        for (size_t t=0; t < zeroPoint.size(); t++)
            zeroPoint[t] = quantParams.getZeroPoint(t);
    }
    return zeroPoint;
}

const mv::Order& mv::Tensor::getInternalOrder() const
{
    return internalOrder_;
}

void mv::Tensor::populateSparsityMapTensor_()
{
    auto shape = getShape();

    std::vector<int64_t> zeroPoint = getZeroPointsPerChannel();
    std::vector<int64_t> sparsityMapData(sparsityMap_->getShape().totalSize());
    std::vector<size_t> sub(shape.ndims());
    uint8_t map = 0;
    std::size_t sparsityMapIdx = 0;
    size_t n = 0;
    int shift = 0;
    int channelIndex = 3;
    auto order = getOrder();

    for (size_t t = 0; t < shape.totalSize(); t++)
    {
        sub = getOrder().indToSub(shape, t);
        if (sub[channelIndex] != n) //starting a new channel, reset map
        {
           if (shift != 0) //this is needed in the case when tensor dimensions are not multiple of 8
            {
                //write map
                sparsityMapData.at(sparsityMapIdx++) = map;
                map = 0;
                shift = 0;
            }
            n = sub[channelIndex];
            if (sparsityMapIdx % 16 != 0)
            {
                auto padding = 16 - (sparsityMapIdx%16);
                sparsityMapIdx += padding;
            }
        }
        if (static_cast<int64_t>(data_[internalOrder_.subToInd(shape, sub)]) != zeroPoint[sub[channelIndex]])
        {
            map += 1 << shift;
            noneZeroElements_++;
        }
        shift++;
        if (shift == 8)//finished one map entry
        {
            sparsityMapData.at(sparsityMapIdx++) = map;
            map = 0;
            shift = 0;
        }
    }
    if (shift != 0)
    {
        //write map
        sparsityMapData.at(sparsityMapIdx++) = map;
    }
    sparsityMap_->populate(sparsityMapData);
}

void mv::Tensor::setAddress(int64_t address)
{
    set<int64_t>("address", address);
    if (isSparse() && !isPopulated())
    {
        auto tensorSize = computeTotalSize();
        auto sparsitySize = sparsityMap_->computeTotalSize();
        storageElement_->set<int64_t>("address", address +
            (tensorSize - storageElement_->computeTotalSize() - sparsitySize));
        sparsityMap_->set<int64_t>("address", address +(tensorSize - sparsitySize));
    }
    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setAddress(address);
}

void mv::Tensor::setSparse()
{
    mv::Order order =  getOrder();
    if (!order.isZMajor())
        throw ArgumentError(*this, "Order", order.toString() , " Sparsity requires ZMajor layout (NHWC)");

    if (order.size() < 3)
        throw ArgumentError(*this, "Order", order.toString() , " Sparsity requires order of size >= 3");

    if(hasAttr("sparse") && get<bool>("sparse") == true)
    //    throw ArgumentError(*this, "Sparsity == ", "true" , " Sparsity for this tensor has already been set");
        return;

    set<bool>("sparse", true);

    // we will create tensors here, and set them as attributes, in runtime_modle, need to check if
    // sparse then get the specific attributes by name and call toBinary
    // this will avoid duplicate of tensors, but it will not allow iterator to go over them.

    //Storage Element Tensor
    //se are filled by the DPU @ runtime
    //We just create an unpopulated tensor at this stage.
    auto shape = getShape();
    size_t N = shape[-1];;

    storageElement_  = std::make_shared<Tensor>(createStorageElementName(getName()), mv::Shape({shape[0], shape[1], 1, N}), mv::DType("Int32"), order);

    set<bool>("sparse", true);

    //Sparsity map
    //we choose layout as internal layout, no need to reorder
    mv::Shape mapShape({shape[0], shape[1], static_cast<std::size_t>(std::ceil(shape[2] / 8.0)), N});
    if (isPopulated())
    {
        //pad the shape
        auto paddedDim = mv::round_up(static_cast<std::size_t>(std::ceil(shape[0] / 8.0)), 16);
        mapShape = {paddedDim, 1, 1, N};
    }
    sparsityMap_ = std::make_shared<Tensor>(createSparsityMapName(getName()), mapShape, mv::DType("UInt8"), Order("NCHW"));
    noneZeroElements_ = 0;

    //populate sparsity map
    if (isPopulated())
        populateSparsityMapTensor_();

    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setSparse();
}

void mv::Tensor::bindData(Tensor& other, const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t> &rightPadding)
{
    if (leftPadding.size() != other.getShape().ndims() && !leftPadding.empty())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Invalid dimension of the left padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(other.getShape().ndims()));

    if (rightPadding.size() != other.getShape().ndims() && !rightPadding.empty())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Invalid dimension of the right padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(getShape().ndims()));

    if (other.isPopulated())
    {
        data_ = other.data_;
        set<bool>("populated", true);
    }

    if (!leftPadding.empty())
        set<std::vector<std::size_t>>("leftPadding", leftPadding);
    if (!rightPadding.empty())
        set<std::vector<std::size_t>>("rightPadding", rightPadding);

    Shape newShape(other.getShape());

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

    if (other.hasAttr("quantizationParams"))
        set<mv::QuantizationParams>("quantizationParams", other.get<mv::QuantizationParams>("quantizationParams"));
}

void mv::Tensor::setOrder(Order order, bool updateSubtensors)
{
    //TODO need to call this with True from DPUTask for weights
    set<Order>("order", order);
    log(Logger::MessageType::Debug, "Reorderd to " + order.toString());
    if (updateSubtensors)
        setSubtensorsOrder_(order);
    return;

}

void mv::Tensor::setSubtensorsOrder_(Order order)
{
    for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        subTensors_[tIdx]->setOrder(order);
}

void mv::Tensor::setDType(DType dtype)
{

    set<DType>("dtype", dtype);
    log(Logger::MessageType::Debug, "Changed data type to " + dtype.toString());
    return;

}

void mv::Tensor::setShape(const Shape& shape)
{
    if(isPopulated())
    {
        log(Logger::MessageType::Warning, "Changing shape of a populated tensor, experimental feature.");
        if(shape.totalSize() != get<Shape>("shape").totalSize())
            throw ArgumentError(*this, "CurrentTensor", "shape", "Unable to change shape of a populated tensor");
    }
    set<Shape>("shape", shape);
    shape_ = shape;
    log(Logger::MessageType::Debug, "Changed shape to " + shape_.toString());
    return;
}


void mv::Tensor::broadcast(const Shape& shape)
{

    if (!isPopulated())
        throw ValueError(*this, "Broadcastring of an unpopulated tensor is undefined");

    if (hasAttr("master") || hasAttr("slave"))
        throw ValueError(*this, "Unable to broadcast a bound tensor");

    Shape s1 = getShape(), s2 = shape;

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
            data_.resize(dataBuf.size(), isDoubleType());

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
            data_[i] = dataBuf[i];
    }

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
std::vector<double> mv::Tensor::getDoubleData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (!isDoubleType())
        throw ValueError(*this, "Attempt of reading double data from an int type tensor");

    std::vector<double> orderedData(getShape().totalSize());

    std::vector<std::size_t> sub(getShape().ndims());
    for (std::size_t i = 0; i < data_.size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(getShape(), i);
            orderedData[getOrder().subToInd(getShape(), sub)] = data_[i];
        }
        else
            orderedData[i] = data_[i];
    }

    return orderedData;
}

std::vector<mv::DataElement> mv::Tensor::getData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    std::vector<DataElement> orderedData(getShape().totalSize(), DataElement(isDoubleType()));

    std::vector<std::size_t> sub(getShape().ndims());
    for (std::size_t i = 0; i < data_.size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(getShape(), i);
            orderedData[getOrder().subToInd(getShape(), sub)] = data_[i];
        }
        else
            orderedData[i] = data_[i];
    }

    return orderedData;
}

std::vector<mv::DataElement> mv::Tensor::getDataPacked()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    auto shape = getShape();
    std::vector<std::size_t> sub(shape.ndims());
    std::vector<int64_t> zeroPoint = getZeroPointsPerChannel();
    std::vector<DataElement> orderedDataPacked;
    double datai;
    size_t outputChannelSize = shape.totalSize() / shape[mv::KERNEL_OUTPUT_CHANNELS];
    for (std::size_t i = 0; i < data_.size(); ++i)
    {
        sub = getOrder().indToSub(getShape(), i);
        datai = data_[internalOrder_.subToInd(getShape(), sub)];
        if (!isSparse() || datai != zeroPoint[sub[mv::KERNEL_OUTPUT_CHANNELS]]) //zero point per output channel
            orderedDataPacked.push_back(DataElement(isDoubleType(), datai));
        //Add padding if needed
        if (isSparse() && ((i+1) % outputChannelSize) == 0) //we reached the end of the outputchannel
        {
            auto size = orderedDataPacked.size() * std::ceil(getDType().getSizeInBits()/8.0);
            auto padsize = mv::round_up(size, 16) - size;
            int64_t zeroPointVal = zeroPoint[sub[mv::KERNEL_OUTPUT_CHANNELS]];
            for (std::size_t j = 0; j < padsize; ++j)
                orderedDataPacked.push_back(DataElement(isDoubleType(), zeroPointVal));
        }
    }
    return orderedDataPacked;
}

std::vector<int64_t> mv::Tensor::getIntData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (isDoubleType())
        throw ValueError(*this, "Attempt of reading int data from an double type tensor");

    std::vector<int64_t> orderedData(getShape().totalSize());

    std::vector<std::size_t> sub(getShape().ndims());
    for (std::size_t i = 0; i < data_.size(); ++i)
    {
        if (getOrder() != internalOrder_)
        {
            sub = internalOrder_.indToSub(getShape(), i);
            orderedData[getOrder().subToInd(getShape(), sub)] = data_[i];
        }
        else
        {
            orderedData[i] = data_[i];
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

bool mv::Tensor::elementWiseChecks_(const Tensor& other)
{
    if (!isPopulated() || !other.isPopulated())
        throw ValueError(*this, "Unable to perfom element-wise operation using unpopulated tensor");

    Shape s1 = getShape(), s2 = other.getShape();

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
        std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(), opFunc);
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
        std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(), opFunc);
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
    if (isDoubleType())
        elementWiseDouble_(other, std::plus<double>());
    else
        elementWiseInt_(other, std::plus<int64_t>());
}

void mv::Tensor::add(double val)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar addition operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_.size(); ++i)
        data_[i] += val;
}

void mv::Tensor::subtract(const Tensor& other)
{
    if (isDoubleType())
        elementWiseDouble_(other, std::minus<double>());
    else
        elementWiseInt_(other, std::minus<int64_t>());

}

void mv::Tensor::subtract(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar subtraction operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_.size(); ++i)
        data_[i] -= val;
}

void mv::Tensor::multiply(const Tensor& other)
{
    if (isDoubleType())
        elementWiseDouble_(other, std::multiplies<double>());
    else
        elementWiseInt_(other, std::multiplies<int64_t>());

}

void mv::Tensor::multiply(double val)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar multiplication operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_.size(); ++i)
        data_[i] *= val;

}

void mv::Tensor::divide(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar division operation for an unpopulated tensor");

    for (unsigned i = 0; i < data_.size(); ++i)
        data_[i] /= val;

}

void mv::Tensor::divide(const Tensor& other)
{
    if (isDoubleType())
        elementWiseDouble_(other, std::divides<double>());
    else
        elementWiseInt_(other, std::divides<int64_t>());

}

void mv::Tensor::sqrt()
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar square root operation for an unpopulated tensor");
    for (unsigned i = 0; i < data_.size(); ++i)
        if (isDoubleType())
            data_[i] = std::sqrt(static_cast<double>(data_[i]));
        else
            data_[i] = std::sqrt(static_cast<int64_t>(data_[i]));

}

mv::DataElement& mv::Tensor::at(const std::vector<std::size_t>& sub)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");
    return data_[subToInd(sub)];
}

const mv::DataElement& mv::Tensor::at(const std::vector<std::size_t>& sub) const
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    return data_[subToInd(sub)];
}

mv::DataElement& mv::Tensor::at(std::size_t idx)
{
    return const_cast<DataElement&>(static_cast<const Tensor*>(this)->at(idx));
}

const mv::DataElement& mv::Tensor::at(std::size_t idx) const
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    if (idx > data_.size())
        throw IndexError(*this, idx, "Exceeds the total lenght of data vector");

    if (hasAttr("master"))
    {
        std::vector<std::size_t> sub = indToSub(idx);
        return data_[subToInd(sub)];
    }

    if (getOrder() == internalOrder_)
        return data_[idx];

    auto sub = getOrder().indToSub(getShape(), idx);
    return data_[internalOrder_.subToInd(getShape(), sub)];
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
    Element::operator=(other);
    data_ = std::vector<DataElement>(other.data_.size(), DataElement(other.isDoubleType()));

    blockSize_ = other.blockSize_;
    blocks_ = std::vector<std::vector<DataElement>::iterator>(other.blocks_.size());
    shape_ = other.shape_;

    for (std::size_t i = 0; i < blocks_.size(); ++i)
        blocks_[i] = data_.begin() + i * blockSize_;

    if (isPopulated())
        data_ = other.data_;

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

mv::BinaryData mv::Tensor::toBinary()
{
    return getDType().toBinary(getDataPacked());
}

std::vector<unsigned> mv::Tensor::computeNumericStrides() const
{
    return getOrder().computeByteStrides(getShape(), getDType().getSizeInBits() / 8);
}

std::size_t mv::Tensor::getClusterSize(bool isBase) const
{
    std::size_t res = computeTotalSize(isBase);
    if (isBroadcasted())
    {
        res = 0;
        for (size_t tIdx = 0; tIdx < subTensors_.size(); tIdx++)
        {
            auto size = subTensors_[tIdx]->computeTotalSize(isBase);
            if (size > res)
                res = size;
        }
    }
    return res;
}

std::size_t mv::Tensor::computeTotalSize(unsigned int alignment, bool isBase) const
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

    if (isSparse())
    {
        if (isPopulated())
        {
            res = noneZeroElements_ * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
        }
        else
        {
            res = shape.totalSize() * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
            res += getSparsityMap()->computeTotalSize();
            res += getStorageElement()->computeTotalSize();
        }
    }
    else
    {
        res = shape.totalSize() * std::ceil(getDType().getSizeInBits()/8.0); //TODO check if we need ceil here?
    }
    //Round up to align to (alignment) 16 bytes
    res = mv::round_up(res, alignment);
    return res;
}

void mv::Tensor::splitAcrossClusters(std::vector<mv::Workload> workloads, bool splitOverH, bool multicast)
{
    if (isPopulated())
    {
        auto shape = getShape();
        for (auto wlItr = workloads.begin(); wlItr != workloads.end(); wlItr++)
        {
            size_t idx = wlItr - workloads.begin();
            if (splitOverH)
            {
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                    getShape(), getDType(), getOrder(), getData()));
            }
            else
            {
                mv::Shape newShape = { shape[0], shape[1] , static_cast<size_t>(wlItr->width), static_cast<size_t>(wlItr->height)};
                auto order = getOrder();
                std::vector<mv::DataElement> splittedData(newShape.totalSize(), mv::DataElement(this->isDoubleType()));
                size_t nOffset = static_cast<size_t>(wlItr->bottomLeftY);
                size_t cOffset = static_cast<size_t>(wlItr->bottomLeftX);
                for (size_t n = 0; n < newShape[3]; n++)
                    for (size_t c = 0; c < newShape[2]; c++)
                        for (size_t h = 0; h < newShape[1]; h++)
                            for (size_t w = 0; w < newShape[0]; w++)
                            {
                                //copy only the relevant channels/kernels
                                splittedData[order.subToInd(newShape, {w, h, c, n})] = this->at({w , h, c+cOffset, n+nOffset});
                            }
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx),
                    newShape, getDType(), order, splittedData));
            }
            std::vector<std::size_t> offset = {0 , 0,
                static_cast<size_t>(wlItr->bottomLeftX), static_cast<size_t>(wlItr->bottomLeftY)};
            subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);

            if (hasAttr("quantizationParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantizationParams", get<mv::QuantizationParams>("quantizationParams"));
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
            if (splitOverH)
            {
                mv::Shape newShape = { static_cast<size_t>(wlItr->width), static_cast<size_t>(wlItr->height) ,shape[2], shape[3]};
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx), newShape, getDType(), getOrder()));
                std::vector<std::size_t> offset = {static_cast<size_t>(wlItr->bottomLeftX), static_cast<size_t>(wlItr->bottomLeftY), 0 , 0};
                subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);
            }
            else
            {
                mv::Shape newShape = { shape[0], shape[1] , static_cast<size_t>(wlItr->width), static_cast<size_t>(wlItr->height)};
                subTensors_.push_back(std::make_shared<mv::Tensor>(getName() + "sub" + std::to_string(idx), newShape, getDType(), getOrder()));
                std::vector<std::size_t> offset =  {0 , 0, static_cast<size_t>(wlItr->bottomLeftX), static_cast<size_t>(wlItr->bottomLeftY)};
                subTensors_[idx]->set<std::vector<std::size_t>>("offset", offset);
            }
            if (hasAttr("quantizationParams"))
                subTensors_[idx]->set<mv::QuantizationParams>("quantizationParams", get<mv::QuantizationParams>("quantizationParams"));
            if (isSparse())
                subTensors_[idx]->setSparse();

            if (!splitOverH || multicast)
            {
                std::vector<std::size_t> leftPadding = subTensors_[idx]->get<std::vector<std::size_t>>("offset");
                std::vector<std::size_t> rightPadding(shape.ndims());
                for (size_t i = 0; i < rightPadding.size(); i++)
                    rightPadding[i] = shape[i] - leftPadding[i] - subTensors_[idx]->getShape()[i];
                subTensors_[idx]->bindData(*this, leftPadding, rightPadding);
            }
        }

        set<bool>("broadcasted", (!splitOverH || multicast));
    }
}
