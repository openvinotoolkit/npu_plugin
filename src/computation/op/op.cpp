#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/computation/model/data_model.hpp"

mv::Op::Op(ComputationModel& model, const std::string& opType, const std::string& name, 
    const std::vector<Data::TensorIterator>& inputs, std::initializer_list<std::pair<std::string, Attribute>> args) :
ModelElement(model, name)
{
    log(Logger::MessageType::Debug, "Initialized");

    if (!op::OpRegistry::checkOpType(opType))
        throw ArgumentError(*this, "opType", opType, "Unregistered op type");
    
    inputs_.reserve(op::OpRegistry::getInputsCount(opType));
    outputs_.reserve(op::OpRegistry::getOutputsCount(opType));

    set<std::string>("opType", opType, {"const"});

    std::cout << "op type is " << opType << std::endl;

    auto argList = op::OpRegistry::argsList(opType); //get list of arguments in ops registry
    auto opsRegistryListSize = argList.size(); //get size of the original ops registry list
    auto argsListWithDefaultValues = op::OpRegistry::argsListWithDefaultValues(opType); //get list of arguments with default values

    for (auto it = args.begin(); it != args.end(); ++it) //for every argument (for the op type) passed to contructor
    {
        auto argIt = std::find(argList.begin(), argList.end(), it->first); // get an iterator to argument (name string) in list of arguments registered get list of arguments in ops registry
                                                                          
        if (argIt != argList.end())                                        // if not at end of the list that matches attribute name in args passed to constructor
        {
            if (!op::OpRegistry::checkArgType(opType, it->first, it->second.getTypeID())) //this is ok - checkArgtype returns bool as checking if elements in both lists match
                throw ArgumentError(*this, "arg", it->first, "Invalid argument type, received " +  
                    attr::AttributeRegistry::getTypeName(it->second.getTypeID()) + ", must be " + 
                    attr::AttributeRegistry::getTypeName(op::OpRegistry::argType(opType, it->first)));
            
            //check if has a default value (not mandatory)
            //if you find the arg in the default args list then it is not mandatory and can be removed from the list
            if(std::find(argsListWithDefaultValues.begin(), argsListWithDefaultValues.end(), *it) != argsListWithDefaultValues.end()) {
                set(it->first, it->second); //set attrbute name and type 
                argList.erase(argIt); //erase it from list of arguments in ops registry
            }
             else {
                set(it->first, it->second); //set attrbute name and type
                
            }
            
            // if not at end of the list that matches attribute name in args passed to constructor
       
        }
        else
            throw ArgumentError(*this, "arg", it->first, "Invalid argument");

    }

    if (argList.size() != (opsRegistryListSize - argsListWithDefaultValues.size())) //check if arg list (mandatory args) is not equal empty
    {
        std::string list;
        for (std::size_t i = 0; i < argList.size() - 1; ++i)
            list += argList[i] + ", ";
        list += argList.back();
        throw ArgumentError(*this, "arg", list, "Missing arguments");
    }

    //this is ok checking inputs only
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
        outputs_[i]->set<std::string>("sourceOp", getName(), {"const"});
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
                    auto destOp = currentFlow.sink();
                    //recursion
                    destOp->setInputTensor(outputs_[i], index);
                }
            }
        }
    }
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

std::vector<mv::Data::TensorIterator> mv::Op::getOutputTensor()
{
    return outputs_;
}

std::size_t mv::Op::inputSlots()
{
    return inputs_.size();
}

std::size_t mv::Op::outputSlots()
{
    return outputs_.size();
}

std::string mv::Op::getLogID() const
{
    return "Op:" + name_;
}
