#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
char mv::Tensor::DataElement::TarrayproxyName[4] = {" PA"};
mv::Tensor::DataElement::DataElement(mv::Tensor* const p /* = 0 */, size_t i /* = 0 */)
{
    if (p) {
        pOwner_ = p;
        idx_ = i;
        strcpy (m_TarrayproxyName, TarrayproxyName);
        TarrayproxyName[2]++;

        std::cout << "    Create DataElement " << m_TarrayproxyName << " idx_ = " << idx_  << std::endl;
    } else {
        throw "DataElement bad p"; //TODO EMAN
    }
}

mv::Tensor::DataElement::~DataElement()
{
        std::cout << "      Destroy TArrayProxy " << m_TarrayproxyName << std::endl;
}

mv::Tensor::DataElement & mv::Tensor::DataElement::operator=(int64_t i)
{
    pOwner_->intData_[idx_] = i;
    //std::cout << "      DataElement assign = i " << i << " to " << pOwner_->m_TarrayName << " using proxy " << m_DataElementName << " idx_ " << idx_ << std::endl;
    return *this;
}
mv::Tensor::DataElement & mv::Tensor::DataElement::operator=(double i)
{
    pOwner_->doubleData_[idx_] = i;
    //std::cout << "      DataElement assign = di " << i << " to " << pOwner_->m_TarrayName << " using proxy " << m_DataElementName << " idx_ " << idx_ << std::endl;
    return *this;
}


mv::Tensor::DataElement & mv::Tensor::DataElement::operator+=(int64_t i)
{
    pOwner_->intData_[idx_] += i;
    //std::cout << "      DataElement add assign += i " << i << " to " << pOwner_->m_TarrayName << " using proxy " << m_DataElementName << " idx_ " << idx_ << std::endl;
    return *this;
}

mv::Tensor::DataElement & mv::Tensor::DataElement::operator+=(double i)
{
    pOwner_->doubleData_[idx_] += i;
    //std::cout << "      DataElement add assign += di " << i << " to " << pOwner_->m_TarrayName << " using proxy " << m_DataElementName << " idx_ " << idx_ << std::endl;
    return *this;
}

// assign an integer value that is specified by a proxy object to a proxy object for a different element.
mv::Tensor::DataElement & mv::Tensor::DataElement::operator = (DataElement src)
{
    //TODO EMAN- assign based on dtype
    if (pOwner_->isDoubleType())
        pOwner_->doubleData_[idx_] = src.pOwner_->doubleData_[src.idx_];
    else
        pOwner_->intData_[idx_] = src.pOwner_->intData_[src.idx_];
    //std::cout << "      DataElement assign = src " << src.m_DataElementName << " idx_ " << src.idx_ << " to " << m_DataElementName << " idx_ "<< idx_ << " from"  << std::endl;
    return *this;
}

mv::Tensor::DataElement::operator int64_t ()
{
    //TODO EMAN change all these to call internal getInt/getDouble
    //which will check if isDoubleType
    if (pOwner_->isDoubleType())
        return pOwner_->doubleData_[idx_];
    return pOwner_->intData_[idx_];
}

mv::Tensor::DataElement::operator double ()
{
    if (pOwner_->isDoubleType())
        return pOwner_->doubleData_[idx_];
    return pOwner_->intData_[idx_];
}
mv::Tensor::DataElement::operator float ()
{
    if (pOwner_->isDoubleType())
        return pOwner_->doubleData_[idx_];
    return pOwner_->intData_[idx_];
}

mv::Tensor::DataElement::operator std::string()
{
    if (pOwner_->isDoubleType())
        return std::to_string(pOwner_->doubleData_[idx_]);
    //else
    return std::to_string(pOwner_->intData_[idx_]);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order):
Element(name),
blockSize_(shape[-1]),
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
    if (isDoubleType())
    {
        doubleData_ = std::vector<double>(shape_.totalSize());
        for (std::size_t i = 0; i < doubleBlocks_.size(); ++i)
            doubleBlocks_[i] = doubleData_.begin() + i * blockSize_;
    }
    else
    {
        intData_ = std::vector<int64_t>(shape_.totalSize());
        for (std::size_t i = 0; i < intBlocks_.size(); ++i)
            intBlocks_[i] = intData_.begin() + i * blockSize_;
    }


}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<double>& data) :
Tensor(name, shape, dType, order)
{
    populate(data, order);
}

mv::Tensor::Tensor(const std::string &name, const Shape &shape, DType dType, Order order, const std::vector<int64_t>& data) :
Tensor(name, shape, dType, order)
{
    if (isDoubleType())
        throw ArgumentError(*this, "data", "int type", "cannot set int data to a double dataType");
    populate(data, order);
}
mv::Tensor::Tensor(const Tensor &other) :
Element(other),
doubleData_(other.doubleData_.size()),
intData_(other.intData_.size()),
blockSize_(other.blockSize_),
doubleBlocks_(other.doubleBlocks_.size()),
intBlocks_(other.intBlocks_.size()),
shape_(other.shape_),
internalOrder_(Order(Order::getRowMajorID(other.shape_.ndims())))
{

    log(Logger::MessageType::Debug, "Copied");
    if (isDoubleType())
    {
        for (std::size_t i = 0; i < doubleBlocks_.size(); ++i)
            doubleBlocks_[i] = doubleData_.begin() + i * blockSize_;
        if (isPopulated())
            doubleData_ = other.doubleData_;
    }
    else
    {
        for (std::size_t i = 0; i < intBlocks_.size(); ++i)
            intBlocks_[i] = intData_.begin() + i * blockSize_;
        if (isPopulated())
            intData_ = other.intData_;
    }
}

mv::Tensor::~Tensor()
{
    log(Logger::MessageType::Debug, "Deleted");
}

mv::Tensor::InternalType mv::Tensor::getInternalType_(DType dtype)
{
    if (dtype == DType("Float8") || dtype == DType("Float16") || dtype == DType("Float32") || dtype == DType("Float64"))
        return Double;
    return Int;
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
        for (std::size_t i = 0; i < doubleData_.size(); ++i)
        {
            sub = getOrder().indToSub(getShape(), i);
            doubleData_[internalOrder_.subToInd(getShape(), sub)] = data[i];
        }
    }
    else
        doubleData_ = data;

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
        for (std::size_t i = 0; i < intData_.size(); ++i)
        {
            sub = getOrder().indToSub(getShape(), i);
            intData_[internalOrder_.subToInd(getShape(), sub)] = data[i];
        }
    }
    else
        intData_ = data;

    set("populated", true);
    log(Logger::MessageType::Debug, "Populated");

    //if sparse then call sparsify
    if (isSparse())
        populateSparsityMapTensor_();
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

void mv::Tensor::unpopulate()
{
    if (!isPopulated())
        return;

    intData_.clear();
    doubleData_.clear();

    set<bool>("populated", false);

    //sparsity map tensor need to unpopulate as well
    if (isSparse())
        sparsityMap_->unpopulate();

    log(Logger::MessageType::Debug, "Unpopulated");
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

std::vector<unsigned> mv::Tensor::getZeroPointsPerChannel_()
{
    //default all zeropoints to zero
    std::vector<unsigned> zeroPoint(getShape()[3]);
    if (isQuantized())
    {
        auto quantParams = get<mv::QuantizationParams>("quantizationParams");
        for (size_t t=0; t < zeroPoint.size(); t++)
            zeroPoint[t] = quantParams.getZeroPoint(t);
    }
    return zeroPoint;
}

void mv::Tensor::populateSparsityMapTensor_()
{
    //default all zeropoints to zero
    std::vector<unsigned> zeroPoint = getZeroPointsPerChannel_();
    std::vector<double> sparsityMapData(sparsityMap_->getShape().totalSize());
    std::vector<size_t> sub(getShape().ndims());
    uint8_t map;
    for (size_t t = 0; t < getShape().totalSize(); t += 8)
    {
        map = 0;
        for (size_t i = 0; i < 8; i++)
        {
            sub = getOrder().indToSub(getShape(), t+i);
            if (isDoubleType())
            {
                if (sub[2] < getShape()[2] && doubleData_[internalOrder_.subToInd(getShape(), sub)] != zeroPoint[sub[3]])
                {
                    map += 1 << i;
                    noneZeroElements_++;
                }
            }
            else
            {
                if (sub[2] < getShape()[2] && intData_[internalOrder_.subToInd(getShape(), sub)] != zeroPoint[sub[3]])
                {
                    map += 1 << i;
                    noneZeroElements_++;
                }
            }
        }
        sparsityMapData[t/8] = map;
    }
    sparsityMap_->populate(sparsityMapData);
}
void mv::Tensor::setSparse()
{
    if (getOrder() != mv::Order("NWHC"))
        throw ArgumentError(*this, "Order", getOrder().toString() , " Sparsity requires ZMajor layout (NWHC)");

    // we will create tensors here, and set them as attributes, in runtime_modle, need to check if
    // sparse then get the specific attributes by name and call toBinary
    // this will avoid duplicate of tensors, but it will not allow iterator to go over them.

    //Storage Element Tensor
    //se are filled by the DPU @ runtime
    //We just create an unpopulated tensor at this stage.
    storageElement_  = std::make_shared<Tensor>(getName() + "_se", mv::Shape({getShape()[0], getShape()[1], 1, getShape()[-1]}), mv::DType("Int32"), getOrder());

    set<bool>("sparse", true);

    //Sparsity map
    //we choose layout as internal layout, no need to reorder
    mv::Shape mapShape({getShape()[0], getShape()[1], static_cast<std::size_t>(std::ceil(getShape()[2] / 8.0)), getShape()[-1]});
    sparsityMap_ = std::make_shared<Tensor>(getName() + "_sm", mapShape, mv::DType("Int8"), Order(Order::getRowMajorID(mapShape.ndims())));
    noneZeroElements_ = 0;

    //populate sparsity map
    if (isPopulated())
    {
        populateSparsityMapTensor_();
    }
}

void mv::Tensor::bindData(Tensor& other, const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t> &rightPadding)
{

    if (!other.isPopulated())
        throw ArgumentError(*this, "InputTensor::populated", "false", "Unable to bind data from an unpopulated tensor " + other.getName());

    if (leftPadding.size() != other.getShape().ndims() && !leftPadding.empty())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Invalid dimension of the left padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(other.getShape().ndims()));

    if (rightPadding.size() != other.getShape().ndims() && !rightPadding.empty())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Invalid dimension of the right padding vector,"
            " must be equal to the dimensionality of the master tensor, which is " + std::to_string(getShape().ndims()));

    intData_ = other.intData_;
    doubleData_ = other.doubleData_;

    set<bool>("populated", true);
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
    other.set<std::string>("slave", getName());

}

void mv::Tensor::setOrder(Order order)
{

    set<Order>("order", order);
    log(Logger::MessageType::Debug, "Reorderd to " + order.toString());
    return;

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
        //TODO EMAN
        std::vector<double> dataBuf = std::vector<double>(sO.totalSize());

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());
        std::cout << " dataBuf.size() " << dataBuf.size() << " doubleData_.size() " << doubleData_.size() << std::endl;
        doubleData_.resize(dataBuf.size());
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
            doubleData_[i] = dataBuf[i]; //TODO EMAN
        std::cout << "Just to get here " << std::endl;

    }
    std::cout << "Just to get here " << std::endl;

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
std::vector<double> mv::Tensor::getDoubleData()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (!isDoubleType())
        throw ValueError(*this, "Attempt of reading double data from an int type tensor");

    std::vector<double> orderedData(getShape().totalSize());
    if (getOrder() == internalOrder_)
        return doubleData_;

    std::vector<std::size_t> sub(getShape().ndims());
    for (std::size_t i = 0; i < doubleData_.size(); ++i)
    {
        sub = internalOrder_.indToSub(getShape(), i);
        orderedData[getOrder().subToInd(getShape(), sub)] = doubleData_[i];
    }

    return orderedData;
}

std::vector<double> mv::Tensor::getDoubleDataPacked()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (!isSparse())
        return getDoubleData();

    std::vector<std::size_t> sub(getShape().ndims());
    std::vector<unsigned> zeroPoint = getZeroPointsPerChannel_();
    std::vector<double> orderedDataPacked;
    double datai;
    orderedDataPacked.reserve(noneZeroElements_);

    for (std::size_t i = 0; i < doubleData_.size(); ++i)
    {
        sub = getOrder().indToSub(getShape(), i);
        datai = doubleData_[internalOrder_.subToInd(getShape(), sub)];
        if (datai != zeroPoint[sub[2]]) //sub[2] is C
            orderedDataPacked.emplace_back(datai);
    }

    return orderedDataPacked;
}

std::vector<int64_t> mv::Tensor::getIntDataPacked()
{
    if (!isPopulated())
        throw ValueError(*this, "Attempt of restoring data from an unpopulated tensor");

    if (!isSparse())
        return getIntData();

    std::vector<std::size_t> sub(getShape().ndims());
    std::vector<unsigned> zeroPoint = getZeroPointsPerChannel_();
    std::vector<int64_t> orderedDataPacked;
    int64_t datai;
    orderedDataPacked.reserve(noneZeroElements_);

    for (std::size_t i = 0; i < intData_.size(); ++i)
    {
        sub = getOrder().indToSub(getShape(), i);
        datai = intData_[internalOrder_.subToInd(getShape(), sub)];
        if (datai != zeroPoint[sub[2]]) //sub[2] is C
            orderedDataPacked.emplace_back(datai);
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
    if (getOrder() == internalOrder_)
        return intData_;
    std::vector<std::size_t> sub(getShape().ndims());
    for (std::size_t i = 0; i < intData_.size(); ++i)
    {
        sub = internalOrder_.indToSub(getShape(), i);
        orderedData[getOrder().subToInd(getShape(), sub)] = intData_[i];
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

void mv::Tensor::elementWiseInt_(const Tensor& other, const std::function<int64_t(int64_t, int64_t)>& opFunc)
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

    if (s1 == s2)
            std::transform(intData_.begin(), intData_.end(), other.intData_.begin(), intData_.begin(), opFunc);
    else
    {

        std::size_t firstIdx = 0;
        while (firstIdx < intBlocks_.size())
        {
            for (std::size_t secondIdx = 0; secondIdx < other.intBlocks_.size(); ++secondIdx)
            {
                std::transform(intBlocks_[firstIdx], intBlocks_[firstIdx] + blockSize_, other.intBlocks_[secondIdx], intBlocks_[firstIdx], opFunc);
                ++firstIdx;
            }
        }

    }

}
void mv::Tensor::elementWiseDouble_(const Tensor& other, const std::function<double(double, double)>& opFunc)
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

    if (s1 == s2)
        std::transform(doubleData_.begin(), doubleData_.end(), other.doubleData_.begin(), doubleData_.begin(), opFunc);
    else
    {

        std::size_t firstIdx = 0;
        while (firstIdx < doubleBlocks_.size())
        {
            for (std::size_t secondIdx = 0; secondIdx < other.doubleBlocks_.size(); ++secondIdx)
            {
                std::transform(doubleBlocks_[firstIdx], doubleBlocks_[firstIdx] + blockSize_, other.doubleBlocks_[secondIdx], doubleBlocks_[firstIdx], opFunc);
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
    if (isDoubleType())
        for (unsigned i = 0; i < doubleData_.size(); ++i)
            doubleData_[i] += val;
    else
        for (unsigned i = 0; i < intData_.size(); ++i)
            intData_[i] += (int64_t) val;
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

    if (isDoubleType())
        for (unsigned i = 0; i < doubleData_.size(); ++i)
            doubleData_[i] -= val;
    else
        for (unsigned i = 0; i < intData_.size(); ++i)
            intData_[i] -= (int64_t) val;

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
    if (isDoubleType())
        for (unsigned i = 0; i < doubleData_.size(); ++i)
            doubleData_[i] *= val;
    else
        for (unsigned i = 0; i < intData_.size(); ++i)
            intData_[i] *= (int64_t) val;

}

void mv::Tensor::divide(double val)
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to perfom scalar division operation for an unpopulated tensor");

    if (isDoubleType())
        for (unsigned i = 0; i < doubleData_.size(); ++i)
            doubleData_[i] /= val;
    else
        for (unsigned i = 0; i < intData_.size(); ++i)
            intData_[i] /= (int64_t) val;


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

    if (isDoubleType())
        for (unsigned i = 0; i < doubleData_.size(); ++i)
            doubleData_[i] = std::sqrt(doubleData_[i]);
    else
        for (unsigned i = 0; i < intData_.size(); ++i)
            intData_[i] = std::sqrt(intData_[i]);

}

mv::Tensor::DataElement mv::Tensor::at(const std::vector<std::size_t>& sub)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");
    return DataElement(this, subToInd(sub));
}

/*const mv::Tensor::DataElement mv::Tensor::at(const std::vector<std::size_t>& sub) const
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    return DataElement(this, (size_t) subToInd(sub));
}*/

mv::Tensor::DataElement mv::Tensor::at(std::size_t idx)
{
    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    if (isDoubleType())
    {
        if (idx > doubleData_.size())
            throw IndexError(*this, idx, "Exceeds the total lenght of data vector");
    }
    else
        if (idx > intData_.size())
            throw IndexError(*this, idx, "Exceeds the total lenght of data vector");

    if (hasAttr("master"))
    {
        std::vector<std::size_t> sub = indToSub(idx);
        return DataElement(this, subToInd(sub));
    }

    if (getOrder() == internalOrder_)
        return DataElement(this, idx);

    auto sub = getOrder().indToSub(getShape(), idx);
    return DataElement(this, internalOrder_.subToInd(getShape(), sub));
    //return const_cast<DataElement>(static_cast<const Tensor*>(this)->at(idx));

}

/*const mv::Tensor::DataElement mv::Tensor::at(std::size_t idx) const
{

    if (!isPopulated())
        throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

    if (isDoubleType())
    {
        if (idx > doubleData_.size())
            throw IndexError(*this, idx, "Exceeds the total lenght of data vector");
    }
    else
        if (idx > intData_.size())
            throw IndexError(*this, idx, "Exceeds the total lenght of data vector");


    if (hasAttr("master"))
    {
        std::vector<std::size_t> sub = indToSub(idx);
        if (isDoubleType())
            return doubleData_[subToInd(sub)];
        else
            return intData_[subToInd(sub)];

    }

    if (getOrder() == internalOrder_)
        if (isDoubleType())
        {
            return doubleData_[idx];
        }
        else
        {
            return intData_[idx];
        }


    auto sub = getOrder().indToSub(getShape(), idx);
    if (isDoubleType())
        return doubleData_[internalOrder_.subToInd(getShape(), sub)];
    //else
    return intData_[internalOrder_.subToInd(getShape(), sub)];
}*/

mv::Tensor::DataElement mv::Tensor::operator()(std::size_t idx)
{
    return at(idx);
}

/*const mv::Tensor::DataElement& mv::Tensor::operator()(std::size_t idx) const
{
    return at(idx);
}*/

mv::Tensor::DataElement mv::Tensor::operator()(const std::vector<std::size_t>& sub)
{
    return at(sub);
}

mv::Tensor& mv::Tensor::operator=(const Tensor& other)
{
    Element::operator=(other);
    intData_ = std::vector<int64_t>(other.intData_.size());
    doubleData_ = std::vector<double>(other.doubleData_.size());

    blockSize_ = other.blockSize_;
    intBlocks_ = std::vector<std::vector<int64_t>::iterator>(other.intBlocks_.size());
    doubleBlocks_ = std::vector<std::vector<double>::iterator>(other.doubleBlocks_.size());
    shape_ = other.shape_;

    for (std::size_t i = 0; i < intBlocks_.size(); ++i)
        intBlocks_[i] = intData_.begin() + i * blockSize_;

    for (std::size_t i = 0; i < doubleBlocks_.size(); ++i)
        doubleBlocks_[i] = doubleData_.begin() + i * blockSize_;

    if (isPopulated())
    {
        intBlocks_ = other.intBlocks_;
        doubleBlocks_ = other.doubleBlocks_;
    }

    return *this;

}

/*const mv::Tensor::DataElement& mv::Tensor::operator()(const std::vector<std::size_t>& sub) const
{
    return at(sub);
}*/

std::string mv::Tensor::getLogID() const
{
    return "Tensor:" + getName();
}
//TODO all toBinaryFunctions should expect the right data
//This should be if else case
mv::BinaryData mv::Tensor::toBinary()
{
    //if (isDoubleType())
    return getDType().toBinary(getDoubleDataPacked());
    //else
    //return getDType().toBinary(getIntDataPacked());
}

std::vector<unsigned> mv::Tensor::computeNumericStrides() const
{
    return getOrder().computeStrides(getShape(), getDType().getSizeInBits() / 8);
}
