#include "include/mcm/computation/op/op.hpp"
#include "include/mcm//op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

mv::Op::Op(ComputationModel& model, const std::string& opType, const std::string& name,
    const std::vector<Data::TensorIterator>& inputs, const std::vector<std::pair<std::string, Attribute>> & args, bool checkInputSize, bool checkArgs) :
ModelElement(model, name)
{
    log(Logger::MessageType::Debug, "Initialized");

    if (!op::OpRegistry::checkOpType(opType))
        throw ArgumentError(*this, "opType", opType, "Unregistered op type");

    inputs_.reserve(op::OpRegistry::getInputsCount(opType));
    outputs_.reserve(op::OpRegistry::getOutputsCount(opType));

    set<std::string>("opType", opType, {"const"});

    auto mandatoryArgList = op::OpRegistry::getArgsList(opType);
    auto numReqdMandatoryArgs = mandatoryArgList.size();
    auto optionalArgList = op::OpRegistry::getOptionalArgsList(opType);

    for (auto it = args.begin(); it != args.end(); ++it)
    {

        auto argIt = std::find(mandatoryArgList.begin(), mandatoryArgList.end(), it->first);

        if (argIt != mandatoryArgList.end())
        {

            if (!op::OpRegistry::checkArgType(opType, it->first, it->second.getTypeID()))
                throw ArgumentError(*this, "arg", it->first, "Invalid argument type, received " +
                    attr::AttributeRegistry::getTypeName(it->second.getTypeID()) + ", must be " +
                    attr::AttributeRegistry::getTypeName(op::OpRegistry::argType(opType, it->first)));

            set(it->first, it->second);

        }
        else
        {
            auto optArgIt = std::find_if(optionalArgList.begin(), optionalArgList.end(),
                                [&it](std::pair<std::string, Attribute> arg)->bool
                                {
                                    return std::get<0>(arg) == it->first;
                                }
                            );

            if (optArgIt != optionalArgList.end())
            {

                if (!op::OpRegistry::checkArgType(opType, it->first, it->second.getTypeID()))
                {

                    throw ArgumentError(*this, "arg", it->first, "Invalid argument type, received " +
                        attr::AttributeRegistry::getTypeName(it->second.getTypeID()) + ", must be " +
                        attr::AttributeRegistry::getTypeName(op::OpRegistry::argType(opType, it->first)));

                }

                set(it->first, it->second);

            }
            else
            {
                if(checkArgs)
                    throw ArgumentError(*this, "arg", it->first, "Invalid argument");
                else
                    set(it->first, it->second);
            }
        }

    }

    // Check if all mandatory args (no default values) provided
    if (args.size() < numReqdMandatoryArgs)
    {
        std::string list;
        for (std::size_t i = 0; i < mandatoryArgList.size() - 1; ++i)
            list += mandatoryArgList[i] + ", ";
        list += mandatoryArgList.back();

        throw ArgumentError(*this, "arg", list, "Missing arguments");
    }

    if(checkInputSize)
        if (inputs.size() != op::OpRegistry::getInputsCount(opType))
            throw ArgumentError(*this, "inputs:size", std::to_string(inputs.size()), "Does not match the registered inputs count " +
                std::to_string(op::OpRegistry::getInputsCount(opType)));

    for (auto it = inputs.begin(); it != inputs.end(); ++it)
        inputs_.push_back(*it);

    std::string errMsg;
    auto checkRes = op::OpRegistry::checkInputs(opType, inputs_, getAttrs_(), errMsg);

    if (!checkRes.first)
        throw OpError(*this, "Invalid input " + op::OpRegistry::getInputLabel(opType, checkRes.second) + " (" +
            std::to_string(checkRes.second) + ") - " + errMsg);

    std::vector<Tensor> outputsDef;
    op::OpRegistry::getOutputsDef(opType, inputs_, getAttrs_(), outputsDef);

    DataModel dm(getModel_());
    for (std::size_t i = 0; i < outputsDef.size(); ++i)
    {
        outputsDef[i].setName(getName() + outputsDef[i].getName());
        outputs_.push_back(dm.defineTensor(outputsDef[i]));
        outputs_[i]->set<std::string>("sourceOp", getName());
    }

    const std::set<std::string>& typeTraits = op::OpRegistry::getTypeTraits(opType);
    std::vector<std::string> opTraits;
    for (auto it = typeTraits.begin(); it != typeTraits.end(); ++it)
        opTraits.push_back(*it);

    set<std::vector<std::string>>("traits", opTraits);
}

mv::Op::~Op()
{
    log(Logger::MessageType::Debug, "Deleted");
}

std::string mv::Op::getOpType() const
{
    return get<std::string>("opType");
}

bool mv::Op::hasTypeTrait(const std::string& typeTrait) const
{
    if (!op::OpRegistry::checkTypeTrait(typeTrait))
        throw ArgumentError(*this, "typeTrait", typeTrait, "Testing against illegal type triat");
    return op::OpRegistry::hasTypeTrait(getOpType(), typeTrait);
}

void mv::Op::redefineOutputTensors()
{
    std::string errMsg;
    auto checkRes = op::OpRegistry::checkInputs(getOpType(), inputs_, getAttrs_(), errMsg);
    if (!checkRes.first)
        throw OpError(*this, "Invalid input " + op::OpRegistry::getInputLabel(getOpType(), checkRes.second) + " (" +
            std::to_string(checkRes.second) + ") - " + errMsg);

    if(hasAttr("invalid") && get<bool>("invalid"))
        erase("invalid");

    std::vector<Tensor> outputsDef;
    op::OpRegistry::getOutputsDef(getOpType(), inputs_, getAttrs_(), outputsDef);
    for (std::size_t i = 0; i < outputsDef.size(); ++i)
    {
        outputs_[i]->setDType(outputsDef[i].getDType());
        outputs_[i]->setOrder(outputsDef[i].getOrder());
        outputs_[i]->setShape(outputsDef[i].getShape());
    }
}

//NOTE: In this function, cascade effect HAS to be handled.
//One could potentially change the input tensor(E.G during a replacement pass) of an operation,
//Causing the output tensor to change as well. But what if the old output tensor was referenced by a flow?

void mv::Op::setInputTensor(Data::TensorIterator tensor, std::size_t idx, bool cascade)
{
    DataModel dm(getModel_());

    //FUTURE: The method should check tensor validity at model level.
    inputs_[idx] = tensor;

    //NOTE: Sometimes we don't want to check the new inputs and generate new outputs
    if(cascade)
    {
        //NOTE: Some operations with multiple inputs such as eltwise, depthwise and concat
        // verify all inputs and weights to be of similar size for specific dimensions.
        // Whenever we update and propagate a tensor change we influence a single
        // input tensor at a time so the input check will go always off.
        // Assume unsafe behavior!
        std::string errMsg;
        auto checkRes = op::OpRegistry::checkInputs(getOpType(), inputs_, getAttrs_(), errMsg);
        if (!checkRes.first)
            throw OpError(*this, "Invalid input " + op::OpRegistry::getInputLabel(getOpType(), checkRes.second) + " (" +
                std::to_string(checkRes.second) + ") - " + errMsg);

        if(hasAttr("invalid") && get<bool>("invalid"))
            erase("invalid");

        std::vector<Tensor> outputsDef;
        op::OpRegistry::getOutputsDef(getOpType(), inputs_, getAttrs_(), outputsDef);
        for (std::size_t i = 0; i < outputsDef.size(); ++i)
        {
            //IDEA: If output tensor definition is updated (keeping the reference)
            //There is no need to do a mess with flaws

            //No need for any outputs_[i]->setName() call, it stays the same;
            outputs_[i]->setDType(outputsDef[i].getDType());
            outputs_[i]->setOrder(outputsDef[i].getOrder());
            outputs_[i]->setShape(outputsDef[i].getShape());

            if(outputs_[i]->hasAttr("flows"))
            {
                //Getting flows in which old output tensor was involved
                auto keys = outputs_[i]->get<std::set<std::string>>("flows");

                //Recursion on the flows
                for(std::string key : keys)
                {
                    auto currentFlow = dm.getDataFlow(key);
                    auto index = currentFlow->get<std::size_t>("sinkInput");
                    currentFlow.sink()->setInputTensor(outputs_[i], index, cascade);
                }
            }
        }
    }
}

// The most brutal method ever: no checks, but only some ops are allowed to do this.
// Obviously there is no cascade effect to handle
unsigned mv::Op::addInputTensor(Data::TensorIterator tensor)
{
    if(mv::op::OpRegistry::checkExtraInputs(getOpType()))
    {
        inputs_.push_back(tensor);
        return inputs_.size();
    }
    else
        throw OpError(*this, "This operation does not support extra inputs after creation");
}

mv::Data::TensorIterator mv::Op::getInputTensor(std::size_t idx)
{
    if (idx >= inputs_.size())
        throw IndexError(*this, idx, "Exceeds the number of inputs");
    return inputs_[idx];
}

mv::Data::TensorIterator mv::Op::getInputTensor(const std::string& label)
{

    const std::vector<std::string>& labels = op::OpRegistry::getInputLabel(getOpType());

    std::size_t idx = 0;
    auto it = labels.begin();
    for (; it != labels.end(); ++it)
    {
        if (*it == label)
            break;
        ++idx;
    }

    if (it == labels.end())
        throw ArgumentError(*this, "input:label", label, "Label not registered for op type " + getOpType());

    return inputs_[idx];

}

std::vector<mv::Data::TensorIterator> mv::Op::getInputTensor()
{
   return inputs_;
}

mv::Data::TensorIterator mv::Op::getOutputTensor(std::size_t idx)
{
     if (idx >= outputs_.size())
        throw IndexError(*this, idx, "Exceeds the number of outputs");
    return outputs_[idx];
}

mv::Data::TensorIterator mv::Op::getOutputTensor(const std::string& label)
{

    const std::vector<std::string>& labels = op::OpRegistry::getOutputLabel(getOpType());

    std::size_t idx = 0;
    auto it = labels.begin();
    for (; it != labels.end(); ++it)
    {
        if (*it == label)
            break;
        ++idx;
    }

    if (it == labels.end())
        throw ArgumentError(*this, "output:label", label, "Label not registered for op type " + getOpType());

    return outputs_[idx];

}

std::map<std::string, mv::Attribute> mv::Op::getAttrs(const std::vector<std::string>& forbiddenKeys) const
{
    std::vector<std::string> forbiddenWords = {"name", "opType", "traits"};
    std::vector<std::string> finalVector(forbiddenKeys);
    finalVector.insert(finalVector.end(), forbiddenWords.begin(), forbiddenWords.end());
    return mv::Element::getAttrs(finalVector);
}

std::vector<mv::Data::TensorIterator> mv::Op::getOutputTensor()
{
    return outputs_;
}

std::size_t mv::Op::inputSlots() const
{
    return inputs_.size();
}

std::size_t mv::Op::outputSlots() const
{
    return outputs_.size();
}

std::string mv::Op::getLogID() const
{
    return "Op:" + name_;
}

bool mv::Op::isImplicit() const
{
    bool isImplicitOp = false;
    std::vector<std::string> implicitTypes = {"ImplicitConcat", "Crop", "Copy", "Slice", "Align", "ImplicitResample", "ImplicitReshape",
                                                "ImplicitPermute", "ImplicitOutput", "ImplicitUnion",
                                                "ImplicitInput", "ImplicitInputSlice", "ImplicitJoin"};
    if (std::count(implicitTypes.begin(), implicitTypes.end(), getOpType()))
    {
        isImplicitOp = true;
    }
    else
        isImplicitOp = false;
    return isImplicitOp;
}

bool mv::Op::isUPA() const
{
    bool isUPATask = false;
    std::vector<std::string> upaTypes = {"Normalize", "Identity", "Softmax", "Proposal", "ROIPooling",
                                        "PSROIPooling", "Resample", "Quantize", "Reshape",
                                        "RegionYolo", "ReorgYolo", "DetectionOutput", "Interp", "Norm",
                                        "Priorbox","Argmax","Permute","CustomOcl","CustomCpp","Sigmoid","Deconv","Tile",
                                        "RefConv", "Gather", "HSwish", "Swish", "Relu", "Conversion", "Tanh", "SoftPlus", "Elu",
                                        "PermuteND", "Mish", "Floor", "Round", "Erf", "Gelu", "Pad", "Interpolate", "MVN",
                                         "Ceiling", "Exp", "SpaceToDepth", "CTCGreedyDecoderSeqLen", "Log", "Prelu", "DepthToSpace", "ReverseSequence", "Power"};
    log(Logger::MessageType::Debug, "isUPA method is called for:" + getOpType());
    if (std::count(upaTypes.begin(), upaTypes.end(), getOpType()))
    {
        isUPATask = true;
    }
    return isUPATask;
}

bool mv::Op::isSparsityConsumer() const
{
    bool isSparseConsumerOp = false;
    const std::vector<std::string> sparseConsumers = {"Conv", "Eltwise"};

    auto opType = getOpType();
    if (std::count(sparseConsumers.cbegin(), sparseConsumers.cend(), opType))
    {
        isSparseConsumerOp = true;
        if (opType == "Eltwise")
        {
            auto opAttrs = getAttrs();
            if (opAttrs.find("eltwiseType") != opAttrs.end())
            {
                if (opAttrs["eltwiseType"].toString() == "\"SqDiff\"")
                    isSparseConsumerOp = false;
            }
        }
    }
    else if (opType == "DPUTask" && std::count(sparseConsumers.cbegin(),
        sparseConsumers.cend(), get<std::string>("taskOp")))
    {
        isSparseConsumerOp = true;
    }
    return isSparseConsumerOp;
}

bool mv::Op::isHardwarizable() const
{
    bool isHardwarizableOp = false;
    std::vector<std::string> hardwarizableTypes =
        {"Conv", "Eltwise", "DepthwiseConv", "MaxPool"};

    auto opType = getOpType();
    if (std::count(hardwarizableTypes.cbegin(), hardwarizableTypes.cend(), opType))
    {
        isHardwarizableOp = true;
        if (opType == "Eltwise")
        {
            auto opAttrs = getAttrs();
            if (opAttrs.find("eltwiseType") != opAttrs.end())
            {
                if (opAttrs["eltwiseType"].toString() == "\"SqDiff\"")
                    isHardwarizableOp = false;
            }
        }
    }
    else if (opType == "DPUTask" && std::count(hardwarizableTypes.cbegin(),
        hardwarizableTypes.cend(), get<std::string>("taskOp")))
    {
        isHardwarizableOp = true;
    }
    return isHardwarizableOp;
}

bool mv::Op::isHwFusable() const
{
    bool isFusableOp = false;
    std::vector<std::string> hwFusableTypes =
        {"Bias", "Sigmoid", "Relu", "LeakyRelu", "Minimum", "Maximum", "Mish"};
    if (std::count(hwFusableTypes.cbegin(), hwFusableTypes.cend(),
        getOpType()))
        isFusableOp = true;
    return isFusableOp;
}

bool mv::Op::hasWeights() const
{
    bool hasWeights = false;
    const std::vector<std::string> weightTypes = {"Conv", "DepthwiseConv"};
    if (std::count(weightTypes.cbegin(), weightTypes.cend(), getOpType()))
        hasWeights = true;
    else if (getOpType() == "DPUTask" &&
        std::count(weightTypes.cbegin(), weightTypes.cend(), get<std::string>("taskOp")))
        hasWeights = true;
    return hasWeights;
}

bool mv::Op::hasPWLActivation() const
{
    const std::vector<std::string> pwlActivations = {
        "Sigmoid",
        "Tanh"
    };
    bool hasPWL = false;
    if (std::find(pwlActivations.cbegin(), pwlActivations.cend(),
        getOpType()) != pwlActivations.cend())
        hasPWL = true;
    if (hasAttr("postOpTypes"))
    {
        for (auto postOp : get<std::vector<std::string>>("postOpTypes"))
        {
            if (std::find(pwlActivations.cbegin(), pwlActivations.cend(),
                postOp) != pwlActivations.cend())
                hasPWL = true;
        }
    }
    return hasPWL;
}

bool mv::Op::hasFloatPrecision() const
{
    return hasAttr("floatPrecision") && get<bool>("floatPrecision");
}

bool mv::Op::supportsCMConv()
{
    OpModel om(getModel_());

    if (hasAttr("supportsCM"))
        return get<bool>("supportsCM");

    if(!(getOpType() == "Conv" && getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] % 16))
        return false;

    std::vector<mv::Data::OpListIterator> ops_in_input_path;
    std::vector<mv::Data::OpListIterator> parents;
    parents.push_back(om.getSourceOp(getInputTensor(0)));
    while (!parents.empty())
    {
        auto opIt = parents.back();
        parents.pop_back();
        if (opIt->isImplicit() || opIt->isUPA() || (opIt->getOpType() == "Bias"))
        {
            // Traverse parent of C-Major-compatible op
            parents.push_back(om.getSourceOp(opIt->getInputTensor(0)));
        }
        else if (opIt->getOpType() == "Concat")
        {
            // Traverse parents of all concat inputs
            for (auto& input : opIt->getInputTensor())
                parents.push_back(om.getSourceOp(input));
        }
        else if (opIt->getOpType() == "Input")
        {
            // Found Input; done traversing this path
        }
        else
        {
            // Found Non C-Major-compatible op
            return false;
        }
        // Store C-Major path op
        ops_in_input_path.push_back(opIt);
    }

    // All traversed paths are C-Major-friendly
    set<bool>("CMinput", true);
    for (auto& op : ops_in_input_path)
        if (op->getOpType() != "Input")
            op->set<bool>("CMinput", true);

    return true;
}
