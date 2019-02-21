#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/base/attribute_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(op::OpRegistry, std::string, mv::op::OpEntry)

}

const std::string mv::op::OpRegistry::compAPIHeaderPath_ = "meta/include/mcm/compositional_model.hpp";
const std::string mv::op::OpRegistry::compAPISourcePath_ = "meta/src/compositional_model.cpp";
const std::string mv::op::OpRegistry::opModelHeaderPath_ = "meta/include/mcm/op_model.hpp";
const std::string mv::op::OpRegistry::opModelSourcePath_ = "meta/src/op_model.cpp";
const std::string mv::op::OpRegistry::recordedCompModelHeaderPath_ = "meta/include/mcm/recorded_compositional_model.hpp";
const std::string mv::op::OpRegistry::recordedCompModelSourcePath_ = "meta/src/recorded_compositional_model.cpp";

/*const std::set<std::string> mv::op::OpRegistry::typeTraits_ = 
{
    "executable",   // An op is doing some processing of inputs
    "exposed"       // An op definition call is exposed in CompositionAPI
};*/

mv::op::OpRegistry::OpRegistry()
{
	typeTraits_.insert("executable");
	typeTraits_.insert("exposed");
}

mv::op::OpRegistry& mv::op::OpRegistry::instance()
{
    
    return Registry<OpRegistry, std::string, OpEntry>::instance();

}

std::vector<std::string> mv::op::OpRegistry::getOpTypes(std::initializer_list<std::string> traits)
{
    auto result = instance().list();
    std::vector<std::string> filteredResults;
    for (auto typeIt = result.begin(); typeIt != result.end(); ++typeIt)
    {
        bool include = true;
        for (auto traitIt = traits.begin(); traitIt != traits.end(); ++traitIt)
        {
            if (!hasTypeTrait(*typeIt, *traitIt))
            {
                include = false;
                break;
            }
        }
        if (include)
            filteredResults.push_back(*typeIt);
    }
    return filteredResults;
    
}

bool mv::op::OpRegistry::checkOpType(const std::string& opType)
{
    return instance().find(opType) != nullptr;
}

bool mv::op::OpRegistry::checkExtraInputs(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining the arguments list for an unregistered op type " + opType);

    OpEntry* const opPtr1 = instance().find(opType);

    if (!opPtr1)
        throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

    return opPtr1->allowsExtraInputs();
}

std::vector<std::string> mv::op::OpRegistry::getArgsList(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining the aandatrguments list for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

    return opPtr->getArgsList();
    
}

//method to return list of attributes with default values and the default values
std::vector<std::pair<std::string, mv::Attribute>> mv::op::OpRegistry::getOptionalArgsList(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining the arguments list for an unregistered op type " + opType);
    
    OpEntry* const opPtr1 = instance().find(opType);

    if (!opPtr1)
        throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

    return opPtr1->getOptionalArgsList();
    
}


std::type_index mv::op::OpRegistry::argType(const std::string& opType, const std::string& argName)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of checking the arguments type for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

    return opPtr->argType(argName);

}

bool mv::op::OpRegistry::checkArgType(const std::string& opType, const std::string& argName, const std::type_index& typeID)
{
    return typeID == argType(opType, argName);
}

std::size_t mv::op::OpRegistry::getInputsCount(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of checking inputs count for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getInputsCount();

}

std::size_t mv::op::OpRegistry::getOutputsCount(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of checking outputs count for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getOutputsCount();

}

std::pair<bool, std::size_t> mv::op::OpRegistry::checkInputs(const std::string& opType,
    const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::string& errMsg)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of executing an inputs check for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->checkInputs(inputs, args, errMsg);
}

void mv::op::OpRegistry::getOutputsDef(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
    const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of executing an inputs check for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    opPtr->getOutputsDef(inputs, args, outputs);

}

std::string mv::op::OpRegistry::getInputLabel(const std::string& opType, std::size_t idx)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining an input label for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getInputLabel(idx);

}

const std::vector<std::string>& mv::op::OpRegistry::getInputLabel(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining an input labels for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getInputLabel();
}

std::string mv::op::OpRegistry::getOutputLabel(const std::string& opType, std::size_t idx)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining an output label for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getOutputLabel(idx);

}

const std::vector<std::string>& mv::op::OpRegistry::getOutputLabel(const std::string& opType)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining an output labels for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getOutputLabel();

}

bool mv::op::OpRegistry::checkTypeTrait(const std::string& typeTrait)
{
    if (instance().typeTraits_.find(typeTrait) != instance().typeTraits_.end())
        return true;
    return false;
}

const std::set<std::string>& mv::op::OpRegistry::getTypeTraits(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining type traits for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    return opPtr->getTypeTraits();
}

bool mv::op::OpRegistry::hasTypeTrait(const std::string& opType, const std::string& trait)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of testing type trait for an unregistered op type " + opType);

    if (!checkTypeTrait(trait))
        throw OpError("OpRegistry", "Attempt of testing against illegal type trait " + trait);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    auto traits = opPtr->getTypeTraits();
    return std::find(traits.begin(), traits.end(), trait) != traits.end();
}

void mv::op::OpRegistry::outputMandatoryArgList(std::vector<std::string>& mandatoryArgsList, OpEntry* const opPtr, std::string& mandatoryArgsDef, bool types)
{
    if (mandatoryArgsList.size() > 0)
    {
        for (std::size_t i = 0; i < mandatoryArgsList.size(); ++i)
        {
            if (types)
            {
                auto attributeName = mandatoryArgsList[i];
                auto argTypeName = attr::AttributeRegistry::getTypeName(opPtr->argType(mandatoryArgsList[i]));
                mandatoryArgsDef += "const " + argTypeName + "& ";
            }

            mandatoryArgsDef += mandatoryArgsList[i];

            if (i < mandatoryArgsList.size() - 1)
                mandatoryArgsDef += ", ";
        }
    }
}

void mv::op::OpRegistry::outputOptionalArgList(std::vector<std::pair<std::string, mv::Attribute>>& optionalArgsList, OpEntry* const opPtr, std::string& optionalArgsDef, bool types, bool defaultArgs)
{
    if (!optionalArgsList.empty())
    {
        for (std::size_t i = 0; i < optionalArgsList.size(); i++)
        {
            if (types)
            {
                auto attributeName = optionalArgsList[i];
                auto argTypeName = attr::AttributeRegistry::getTypeName(opPtr->argType(optionalArgsList[i].first));
                optionalArgsDef += "const " + argTypeName + "& " + optionalArgsList[i].first;
                if (defaultArgs)
                    optionalArgsDef += " = " + optionalArgsList[i].second.toString();
            }
            else
                optionalArgsDef += optionalArgsList[i].first;


            if (i < optionalArgsList.size() - 1)
                optionalArgsDef += ", ";
        }
    }
}

std::string mv::op::OpRegistry::getCompositionDeclSig_(const std::string& opType, bool args, bool types, bool defaultArgs, bool call, bool recordedModel)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining CompositionAPI declaration for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);
    bool inputVectorTypes = opPtr->hasVectorTypesAsInput();

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    if (opPtr->getOutputsCount() > 1)
        throw MasterError("OpRegistry", "Multi-output ops currently unsupported in CompositionAPI generator");

    std::string output;

    auto copyOps = opPtr->getCopyOperations();
    if(copyOps.size() == 0)
        copyOps.push_back(opType);

    //One signature for each copied operation signature
    for(auto vecIt = copyOps.begin(); vecIt != copyOps.end(); ++vecIt)
    {
        std::string copyOp(*vecIt);
        OpEntry* const copyOpPtr = instance().find(copyOp);

        if(!call)
        {
            output += "mv::Data::TensorIterator ";
            if(recordedModel)
                output += "mv::RecordedCompositionalModel:: ";
        }

        output.push_back(std::tolower(opType[0]));
        output += opType.substr(1);

        if(opType != copyOp)
            output += copyOp;

        if (args)
        {

            output += "(";
            std::string inputsDef = "";
            auto inputLabels = opPtr->getInputLabel();
            if (inputLabels.size() > 0)
            {
                for (std::size_t i = 0; i < inputLabels.size() - 1; ++i)
                {
                    if (types)
                    {
                        if (inputVectorTypes)
                            inputsDef += "const std::vector< ";
                        inputsDef += "Data::TensorIterator ";
                        if(inputVectorTypes)
                            inputsDef += ">& ";
                    }
                    inputsDef += inputLabels[i] + ", ";
                }

                if (types)
                {
                    if (inputVectorTypes)
                        inputsDef += "const std::vector< ";
                    inputsDef += "Data::TensorIterator ";
                    if(inputVectorTypes)
                        inputsDef += ">& ";
                }
                inputsDef += inputLabels.back();
            }

            std::string mandatoryArgsDef = "";
            std::string optionalArgsDef = "";
            auto mandatoryArgsList = opPtr->getArgsList();
            auto optionalArgsList = opPtr->getOptionalArgsList(); /*Get arg list with default values*/

            auto copyOpMandatoryArgsList = copyOpPtr->getArgsList();
            auto copyOptionalArgsList = copyOpPtr->getOptionalArgsList();

            if(copyOp == opType)
            {
                copyOpMandatoryArgsList.clear();
                copyOptionalArgsList.clear();
            }

            std::string defaultValue = "";

            if(!copyOpMandatoryArgsList.empty())
                outputMandatoryArgList(copyOpMandatoryArgsList, copyOpPtr, mandatoryArgsDef, types);
            else
                outputMandatoryArgList(mandatoryArgsList, opPtr, mandatoryArgsDef, types);

            if(!copyOptionalArgsList.empty())
                outputOptionalArgList(copyOptionalArgsList, copyOpPtr, optionalArgsDef, types, defaultArgs);
            else
                outputOptionalArgList(optionalArgsList, opPtr, optionalArgsDef, types, defaultArgs);

            output += inputsDef;
            if (!inputsDef.empty())
                output += ", ";

            output += mandatoryArgsDef;
            if (!mandatoryArgsDef.empty())
                output += ", ";

            output += optionalArgsDef;
            if (!optionalArgsDef.empty())
                output += ", ";

            if (types)
                output += "const std::string& ";
            output += "name";

            if (defaultArgs)
                output += " = \"\"";

            output += ")";

            auto backupIt = vecIt;
            if(++backupIt != copyOps.end())
                output += ";\n";

        }
    }

    return output;
}

std::string mv::op::OpRegistry::getCompositionDecl_(const std::string& opType)
{
    return getCompositionDeclSig_(opType, true, true, true, false, false);
}

std::string mv::op::OpRegistry::getCompositionCall_(const std::string& opType)
{
    return getCompositionDeclSig_(opType, true, false, false, true, false);
}

std::string mv::op::OpRegistry::getLabelNameStringifyCall_(const std::string& label, const std::string& name, std::size_t idx, 
    const std::string& indent, const std::string& eol)
{
    return indent + "std::string " + name + std::to_string(idx) + " = " + label + "->getName();" + eol +
        indent + "std::transform(" + name + std::to_string(idx) + ".begin()," +
        name +  std::to_string(idx) + ".end(), " + name + std::to_string(idx) + ".begin(), ::tolower);" + eol
        + indent + "std::replace(" + name + std::to_string(idx) + ".begin(), " + name + std::to_string(idx) + 
        ".end(), ':', '_');" + eol;
}

std::string mv::op::OpRegistry::getStringifiedInputsCall_(const std::string opType, const std::string& indent,
    const std::string eol)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining a stringified inputs representation call for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    std::string output = "";
    auto inputLabels = opPtr->getInputLabel();
    if (inputLabels.size() > 0)
    {
        for (std::size_t i = 0; i < inputLabels.size() - 1; ++i)
            output += getLabelNameStringifyCall_(inputLabels[i], "input", i, indent, eol);
        
        output += getLabelNameStringifyCall_(inputLabels.back(), "input", inputLabels.size() - 1, indent, eol);
        
    }

    return output;

}

std::string mv::op::OpRegistry::getStringifiedOutputsCall_(const std::string opType, const std::string& indent, const std::string eol)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining a stringified outputs representation call for an unregistered op type " + opType);

    if (getOutputsCount(opType) > 1)
        throw MasterError("OpRegistry", "Generation of stringifed label names call is currently unsupported for mulit-output ops");

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    std::string output = "";
    auto outputLabels = opPtr->getOutputLabel();
    if (outputLabels.size() > 0)
    {
        for (std::size_t i = 0; i < outputLabels.size() - 1; ++i)
            output += getLabelNameStringifyCall_(outputLabels[i], "output", i, indent, eol);
        
        output += getLabelNameStringifyCall_(outputLabels.back(), "output", outputLabels.size() - 1, indent, eol);
        
    }

    return output;

}

std::vector<std::string> mv::op::OpRegistry::getStringifiedArgsCall_(const std::string opType)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining a stringified args representation call for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    std::vector<std::string> output;
    auto argsList = opPtr->getArgsList();
    if (argsList.size() > 0)
    {
        for (std::size_t i = 0; i < argsList.size() - 1; ++i)
            output.push_back("Attribute(" + argsList[i] + ").toLongString()");
        output.push_back("Attribute(" + argsList.back() + ").toLongString()");
    }

    return output;

}

void mv::op::OpRegistry::defineOpOutput(std::string& output, const std::string& eol, const std::string& opType, OpEntry* const opPtr, std::string token, bool inputVectorTypes, bool checkInputs, bool copiedOp, const std::string& tab)
{
    output += token + eol + "{" +
        eol + tab + "return defineOp(" + eol + tab + tab + "\"" + opType + "\"," + eol + tab + tab;
    if(!inputVectorTypes)
        output += "{";

    auto inputLabels = opPtr->getInputLabel();
    if(inputVectorTypes)
        output += "inputs";
    else if (inputLabels.size() > 0)
    {
        for (std::size_t i = 0; i < inputLabels.size() - 1; ++i)
            output +=  eol + tab + tab + tab + inputLabels[i] + ",";
        output +=  eol + tab + tab + tab + inputLabels.back();
    }
    output += eol + tab + tab;
    if(!inputVectorTypes)
        output += + "}";
    output += "," + eol + tab + tab;
    output += "{";

    auto mandatoryArgsList = opPtr->getArgsList();
    if(copiedOp)
        output +=  eol + tab + tab + tab + "{ \"taskOp\", std::string(\"" + opPtr->getName() + "\") },";

    if (mandatoryArgsList.size() > 0)
    {
        for (std::size_t i = 0; i < mandatoryArgsList.size() - 1; ++i)
            output +=  eol + tab + tab + tab + "{ \"" + mandatoryArgsList[i] + "\", " + mandatoryArgsList[i] + " },";
        output +=  eol + tab + tab + tab + "{ \"" + mandatoryArgsList.back() + "\", " + mandatoryArgsList.back() + " }";
    }

    auto optionalArgsList = opPtr->getOptionalArgsList();
    if (optionalArgsList.size() > 0)
    {
        if (mandatoryArgsList.size() > 0)
            output += ",";

        for (std::size_t i = 0; i < optionalArgsList.size() - 1; ++i)
            output +=  eol + tab + tab + tab + "{ \"" + optionalArgsList[i].first + "\", " + optionalArgsList[i].first + " },";
        output +=  eol + tab + tab + tab + "{ \"" + optionalArgsList.back().first + "\", " + optionalArgsList.back().first + " }";
    }
    output += eol + tab + tab;

    output += "}," + eol + tab + tab + "name";

    if(inputVectorTypes)
        output += ",";
    output += eol + tab;
    if(inputVectorTypes)
        output += tab + "false";

    if(!checkInputs)
        output += ",";
    output += eol + tab;
    if(!checkInputs)
        output += tab + "false";

    output += eol + tab;

    output += ");";
    output += eol + "}";
}

std::string mv::op::OpRegistry::getCompositionDef_(const std::string& opType, const std::string& eol, const std::string& tab)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining CompositionAPI definition for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);
    auto copiedOps = opPtr->getCopyOperations();

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    if (opPtr->getOutputsCount() > 1)
        throw MasterError("OpRegistry", "Multi-output ops currently unsupported in CompositionAPI generator");

    bool inputVectorTypes = opPtr->hasVectorTypesAsInput();
    bool checkInputs = opPtr->doInputNeedToBeChecked();

    std::string signatures = getCompositionDeclSig_(opType, true, true, false, true, false);
    std::string delimiter = ";";

    std::string output = "";

    bool isCopiedOpsEmpty = copiedOps.empty();

    size_t pos = 0;
    std::string token;
    size_t copiedOpsIndex = 0;
    //Signatures and Copied operations are in the same order
    while ((pos = signatures.find(delimiter)) != std::string::npos)
    {
        token = signatures.substr(0, pos);
        OpEntry* const copiedPpPtr = instance().find(copiedOps[copiedOpsIndex++]);
        output += "mv::Data::TensorIterator mv::OpModel::";
        defineOpOutput(output, eol, opType, copiedPpPtr, token, inputVectorTypes, checkInputs, !isCopiedOpsEmpty, tab);

        signatures.erase(0, pos + delimiter.length());
    }

    output += "mv::Data::TensorIterator mv::OpModel::";
    if(isCopiedOpsEmpty)
        defineOpOutput(output, eol, opType, opPtr, signatures, inputVectorTypes, checkInputs, !isCopiedOpsEmpty, tab);
    else
    {
        OpEntry* const copiedPpPtr = instance().find(copiedOps[copiedOpsIndex]);
        defineOpOutput(output, eol, opType, copiedPpPtr, signatures, inputVectorTypes, checkInputs, !isCopiedOpsEmpty, tab);
    }

    return output;

}

void mv::op::OpRegistry::generateCompositionAPI(const std::string& eol, const std::string& tab)
{

    std::ofstream incStream(utils::projectRootPath() + "/" + compAPIHeaderPath_, std::ios::out | std::ios::trunc);
    if (!incStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the CompositionalModel header file during the CompositionAPI generation");

    incStream << "/*" << eol;
    incStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    incStream << "*/" << eol << eol;

    incStream << "#ifndef MV_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream << "#define MV_COMPOSITIONAL_MODEL_HPP_" << eol << eol;
    incStream << "#include \"include/mcm/computation/model/iterator/data_context.hpp\"" << eol;
    incStream << "#include \"include/mcm/computation/model/iterator/tensor.hpp\"" << eol << eol;

    incStream << "namespace mv" << eol << eol;
    incStream << "{" << eol << eol;
    incStream << tab << "class CompositionalModel" << eol;
    incStream << tab << "{" << eol << eol;
    incStream << tab << "public:" << eol << eol;

    incStream << tab << tab << "virtual ~CompositionalModel() = 0;" << eol << eol;

    auto exposedOpsList = getOpTypes({"exposed"});
    for (auto it = exposedOpsList.begin(); it != exposedOpsList.end(); ++it)
        incStream << tab << tab << "virtual " + getCompositionDecl_(*it) << " = 0;" << eol;

    incStream << eol << tab << tab << "virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;" << eol;
    incStream << tab << tab << "virtual void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) = 0;" << eol;
    incStream << tab << tab << "virtual bool isValid() const = 0;" << eol;
    incStream << tab << tab << "virtual bool isValid(Data::TensorIterator tensor) const = 0;" << eol;
    incStream << tab << tab << "virtual bool isValid(Data::OpListIterator op) const = 0;" << eol;
    incStream << tab << tab << "virtual std::string getName() const = 0;" << eol << eol;

    incStream << tab << "};" << eol << eol;
    incStream << "}" << eol << eol;

    incStream << "#endif //MV_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream.close();

    std::ofstream srcStream(utils::projectRootPath() + "/" + compAPISourcePath_, std::ios::out | std::ios::trunc);
    if (!srcStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the CompositionalModel source file during the CompositionAPI generation");

    srcStream << "/*" << eol;
    srcStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    srcStream << "*/" << eol << eol;

    srcStream << "#include \"" << compAPIHeaderPath_ << "\"" << eol << eol;
    srcStream << "mv::CompositionalModel::~CompositionalModel()" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;
    srcStream.close();

    incStream.open(utils::projectRootPath() + "/" + opModelHeaderPath_, std::ios::out | std::ios::trunc);
    if (!incStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the OpModel header file during the CompositionAPI generation");

    incStream << "/*" << eol;
    incStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    incStream << "*/" << eol << eol;

    incStream << "#ifndef MV_OP_MODEL_HPP_" << eol;
    incStream << "#define MV_OP_MODEL_HPP_" << eol << eol; 
    incStream << "#include \"" << compAPIHeaderPath_ << "\"" << eol;
    incStream << "#include \"include/mcm/computation/model/base_op_model.hpp\"" << eol << eol;

    incStream << "namespace mv" << eol << eol;
    incStream << "{" << eol << eol;
    incStream << tab << "class OpModel: public BaseOpModel, public CompositionalModel" << eol;
    incStream << tab << "{" << eol << eol;
    incStream << tab << "public:" << eol << eol;
    incStream << tab << tab << "OpModel(const std::string& name);" << eol;
    incStream << tab << tab << "OpModel(ComputationModel& model);" << eol;
    incStream << tab << tab << "virtual ~OpModel();" << eol << eol;

    auto opsList = getOpTypes({});
    for (auto it = opsList.begin(); it != opsList.end(); ++it)
    {
        incStream << tab << tab << getCompositionDecl_(*it);
        if (hasTypeTrait(*it, "exposed"))
            incStream << " override";
        incStream << ";" << eol;
    }
    incStream << eol << tab << tab << "Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;" << eol;
    incStream << tab << tab << "void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;" << eol;
    incStream << tab << tab << "bool isValid() const override;" << eol;
    incStream << tab << tab << "bool isValid(Data::TensorIterator tensor) const override;" << eol;
    incStream << tab << tab << "bool isValid(Data::OpListIterator op) const override;" << eol;
    incStream << tab << tab << "std::string getName() const override;" << eol << eol;

    incStream << tab << "};" << eol << eol;
    incStream << "}" << eol << eol;

    incStream << "#endif //MV_OP_MODEL_HPP_" << eol;
    incStream.close();

    srcStream.open(utils::projectRootPath() + "/" + opModelSourcePath_, std::ios::out | std::ios::trunc);
    if (!srcStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the OpModel source file during the CompositionAPI generation");

    srcStream << "/*" << eol;
    srcStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    srcStream << "*/" << eol << eol;
    srcStream << "#include \"" << opModelHeaderPath_ << "\"" << eol << eol;

    srcStream << "mv::OpModel::OpModel(const std::string& name) :" << eol;
    srcStream << "BaseOpModel(name)" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;
    srcStream << "mv::OpModel::OpModel(ComputationModel& other) :" << eol;
    srcStream << "BaseOpModel(other)" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;
    srcStream << "mv::OpModel::~OpModel()" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;

    for (auto it = opsList.begin(); it != opsList.end(); ++it)
        srcStream << getCompositionDef_(*it, eol, tab) << eol << eol;
    
    srcStream << "mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::getSourceOp(tensor);" << eol;
    srcStream << "}" << eol;
    srcStream << "void mv::OpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::addAttr(op, name, attr);" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::OpModel::isValid() const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::isValid();" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::OpModel::isValid(Data::TensorIterator tensor) const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::isValid(tensor);" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::OpModel::isValid(Data::OpListIterator op) const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::isValid(op);" << eol;
    srcStream << "}" << eol;
    srcStream << "std::string mv::OpModel::getName() const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return BaseOpModel::getName();" << eol;
    srcStream << "}" << eol;
    srcStream.close();

}

void mv::op::OpRegistry::generateRecordedCompositionAPI(const std::string& eol, const std::string& tab)
{

    std::ofstream incStream(utils::projectRootPath() + "/" + recordedCompModelHeaderPath_, std::ios::out | std::ios::trunc);
    if (!incStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the header file during the RecordedCompositionAPI generation");

    incStream << "/*" << eol;
    incStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateRecordedCompositionAPI()" << eol;
    incStream << "*/" << eol << eol;

    incStream << "#ifndef MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream << "#define MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol << eol; 
    incStream << "#include <fstream>" << eol;
    incStream << "#include \"" + compAPIHeaderPath_ + "\"" << eol << eol;

    incStream << "namespace mv" << eol << eol;
    incStream << "{" << eol << eol;
    incStream << tab << "class RecordedCompositionalModel : public CompositionalModel" << eol;
    incStream << tab << "{" << eol << eol;
    incStream << tab << tab << "CompositionalModel &model_;" << eol;
    incStream << tab << tab << "std::ofstream srcStream_;" << eol;
    incStream << tab << tab << "std::string tab_;" << eol << eol;
    incStream << tab << "public:" << eol << eol;

    incStream << tab << tab << "RecordedCompositionalModel(CompositionalModel& model,"
        " const std::string& outputPath, const std::string tab = \"    \");" << eol;
    incStream << tab << tab << "virtual ~RecordedCompositionalModel();" << eol << eol;

    auto opsList = getOpTypes({"exposed"});
    for (auto it = opsList.begin(); it != opsList.end(); ++it)
        incStream << tab << tab << getCompositionDecl_(*it) << " override;" << eol;

    incStream << eol << tab << tab << "Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;" << eol;
    incStream << tab << tab << "void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;" << eol;
    incStream << tab << tab << "bool isValid() const override;" << eol;
    incStream << tab << tab << "bool isValid(Data::TensorIterator tensor) const override;" << eol;
    incStream << tab << tab << "bool isValid(Data::OpListIterator op) const override;" << eol;
    incStream << tab << tab << "std::string getName() const override;" << eol << eol;

    incStream << tab << "};" << eol << eol;
    incStream << "}" << eol << eol;

    incStream << "#endif //MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream.close();

    std::ofstream srcStream(utils::projectRootPath() + "/" + recordedCompModelSourcePath_, std::ios::out | std::ios::trunc);
    if (!srcStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the source file during the CompositionAPI generation");

    srcStream << "/*" << eol;
    srcStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    srcStream << "*/" << eol << eol;

    srcStream << "#include \"" << recordedCompModelHeaderPath_ << "\"" << eol << eol;

    srcStream << "mv::RecordedCompositionalModel::RecordedCompositionalModel(CompositionalModel& model,"
        " const std::string& outputPath, const std::string tab) :" << eol;
    srcStream << tab << "model_(model)," << eol;
    srcStream << tab << "srcStream_(outputPath + model_.getName() + \".cpp\", std::ios::out | std::ios::trunc)," << eol;
    srcStream << tab << "tab_(tab)" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "srcStream_ << \"#include \\\"" << utils::projectRootPath() << "\" << \"/meta/include/mcm/compositional_model.hpp\\\"\""
        "<< std::endl;" << eol;
    srcStream << tab << "srcStream_ << \"#include \\\"" << utils::projectRootPath() << "\" << \"/include/mcm/computation/model/op_model.hpp\\\"\""
        "<< std::endl;" << eol; 
    srcStream << tab << "srcStream_ << std::endl << \"int main()\" << std::endl << \"{\" << std::endl << tab_ << \"using namespace mv;\" << std::endl;" << eol;
    srcStream << tab << "srcStream_ << tab_ << \"OpModel model(\\\"\" << getName() << \"\\\");\" << std::endl;" << eol;
    srcStream << "}" << eol << eol;

    srcStream << "mv::RecordedCompositionalModel::~RecordedCompositionalModel()" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "srcStream_ << std::endl << tab_ << \"return 0;\" << std::endl << \"}\" << std::endl;" << eol;
    srcStream << tab << "srcStream_.close();" << eol;
    srcStream << "}" << eol << eol;

    for (auto it = opsList.begin(); it != opsList.end(); ++it)
    {

        srcStream << getCompositionDeclSig_(*it, true, true, false, false, true)
            << eol << "{" << eol;
        
        srcStream << tab << "Data::TensorIterator output = model_." << getCompositionCall_(*it) << ";" << eol;
        // That assumes one output, to be changed when multi-output ops are supported
        
        std::string inputNameDefs = getStringifiedInputsCall_(*it, tab, eol);
        std::string outputNameDefs = getStringifiedOutputsCall_(*it, tab, eol);
        srcStream << inputNameDefs;
        srcStream << outputNameDefs; 
        srcStream << tab << "srcStream_ << tab_";
        if (!outputNameDefs.empty())
        {
            srcStream << " << \"Data::TensorIterator \" + output0 + \" = \"";
        }

        srcStream << " << \"model." << getCompositionDeclSig_(*it, false, false, false, true, true) << "(\" << ";

        
        if (getInputsCount(*it) > 0)
        {
            for (std::size_t i = 0; i < getInputsCount(*it) - 1; ++i)
                srcStream << "input" + std::to_string(i) << "<< \", \" << ";
            srcStream << "input" + std::to_string(getInputsCount(*it) - 1);
        }

        auto argsStr = getStringifiedArgsCall_(*it);
        if (argsStr.size() > 0)
        {
            if (getInputsCount(*it) > 0)
                srcStream << " << \", \" << ";
            for (std::size_t i = 0; i < argsStr.size() - 1; ++i)
                srcStream << argsStr[i] << " << \", \" << ";
            srcStream << argsStr.back(); 
        }

        srcStream << " << \");\" << std::endl;" << eol;

        srcStream << tab << "return output;" << eol << "}" << eol;
    
    }

    srcStream << "mv::Data::OpListIterator mv::RecordedCompositionalModel::getSourceOp(Data::TensorIterator tensor)" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.getSourceOp(tensor);" << eol;
    srcStream << "}" << eol;
    srcStream << "void mv::RecordedCompositionalModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.addAttr(op, name, attr);" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::RecordedCompositionalModel::isValid() const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.isValid();" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::RecordedCompositionalModel::isValid(Data::TensorIterator tensor) const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.isValid(tensor);" << eol;
    srcStream << "}" << eol;
    srcStream << "bool mv::RecordedCompositionalModel::isValid(Data::OpListIterator op) const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.isValid(op);" << eol;
    srcStream << "}" << eol;
    srcStream << "std::string mv::RecordedCompositionalModel::getName() const" << eol;
    srcStream << "{" << eol;
    srcStream << tab << "return model_.getName();" << eol;
    srcStream << "}" << eol;
    srcStream.close();
    
}
