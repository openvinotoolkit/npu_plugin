#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/base/attribute_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(std::string, mv::op::OpEntry)

}

const std::string mv::op::OpRegistry::compAPIHeaderPath_ = "include/mcm/api/compositional_model.hpp";
const std::string mv::op::OpRegistry::compAPISourcePath_ = "src/api/compositional_model.cpp";
const std::string mv::op::OpRegistry::recordedCompAPIHeaderPath_ = "include/mcm/utils/recorded_compositional_model.hpp";
const std::string mv::op::OpRegistry::recordedCompAPISourcePath_ = "src/utils/recorded_compositional_model.cpp";

const std::set<std::string> mv::op::OpRegistry::typeTraits_ = 
{
    "executable",   // An op is doing some processing of inputs
    "exposed"       // An op definition call is exposed in CompositionAPI
};

mv::op::OpRegistry& mv::op::OpRegistry::instance()
{
    
    return static_cast<OpRegistry&>(Registry<std::string, OpEntry>::instance());

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

std::vector<std::string> mv::op::OpRegistry::argsList(const std::string& opType)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining the arguments list for an unregistered op type " + opType);
    
    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType + " not found in the op registry");

    return opPtr->argsList();
    
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
    if (typeTraits_.find(typeTrait) != typeTraits_.end())
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

std::string mv::op::OpRegistry::getCompositionDeclSig_(const std::string& opType, bool types, bool defaultArgs)
{
    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining CompositionAPI declaration for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    if (!opPtr->hasTypeTrait("exposed"))
        throw OpError("OpRegistry", "Call for CompositionAPI declaration generation for a non-exposed op type " + opType);

    if (opPtr->getOutputsCount() > 1)
        throw MasterError("OpRegistry", "Multi-output ops currently unsupported in CompositionAPI generator");

    std::string output;
    output.push_back(std::tolower(opType[0]));
    output += opType.substr(1) + "(";
    
    std::string inputsDef = "";
    auto inputLabels = opPtr->getInputLabel();
    if (inputLabels.size() > 0)
    {
        for (std::size_t i = 0; i < inputLabels.size() - 1; ++i)
        {
            if (types)
                inputsDef += "Data::TensorIterator ";
            inputsDef += inputLabels[i] + ", ";
        }

        if (types)
             inputsDef += "Data::TensorIterator ";
        inputsDef += inputLabels.back();
    }

    std::string argsDef = "";
    auto argsList = opPtr->argsList();
    if (argsList.size() > 0)
    {
        for (std::size_t i = 0; i < argsList.size() - 1; ++i)
        {
            if (types)
            {
                auto argTypeName = attr::AttributeRegistry::getTypeName(opPtr->argType(argsList[i]));
                argsDef += "const " + argTypeName + "& ";
            }    
            
            argsDef += argsList[i] + ", ";
        }

        if (types)
        {
            auto argTypeName = attr::AttributeRegistry::getTypeName(opPtr->argType(argsList.back()));
            argsDef += "const " + argTypeName + "& ";
        }
        argsDef += argsList.back();

    }

    output += inputsDef;
    if (!inputsDef.empty())
        output += ", ";

    output += argsDef;
    if (!argsList.empty())
        output += ", ";

    if (types)
        output += "const std::string&";
    output += "name";
    
    if (defaultArgs)
        output += " = \"\"";

    return output + ")";
}

std::string mv::op::OpRegistry::getCompositionDecl_(const std::string& opType)
{
    return "Data::TensorIterator " + getCompositionDeclSig_(opType, true, true);
}

std::string mv::op::OpRegistry::getCompositionCall_(const std::string& opType)
{
    return "CompositionalModel::" + getCompositionDeclSig_(opType, false, false);
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
    auto argsList = opPtr->argsList();
    if (argsList.size() > 0)
    {
        for (std::size_t i = 0; i < argsList.size() - 1; ++i)
            output.push_back("Attribute(" + argsList[i] + ").toString()");
        output.push_back("Attribute(" + argsList.back() + ").toString()");
    }

    return output;

}

std::string mv::op::OpRegistry::getCompositionDef_(const std::string& opType, const std::string& eol, const std::string& tab)
{

    if (!checkOpType(opType))
        throw OpError("OpRegistry", "Attempt of obtaining CompositionAPI definition for an unregistered op type " + opType);

    OpEntry* const opPtr = instance().find(opType);

    if (!opPtr)
        throw MasterError("OpRegistry", "Registered op type " + opType +
            " not found in the op registry");

    if (!opPtr->hasTypeTrait("exposed"))
        throw OpError("OpRegistry", "Call for CompositionAPI definition generation for a non-exposed op type " + opType);

    if (opPtr->getOutputsCount() > 1)
        throw MasterError("OpRegistry", "Multi-output ops currently unsupported in CompositionAPI generator");

    std::string output = getCompositionDeclSig_(opType, true, false) + eol + "{" + 
        eol + tab + "return defineOp(" + eol + tab + tab + "\"" + opType + "\"," + eol + tab + tab + "{";
    
    auto inputLabels = opPtr->getInputLabel();
    if (inputLabels.size() > 0)
    {
        for (std::size_t i = 0; i < inputLabels.size() - 1; ++i)
            output +=  eol + tab + tab + tab + inputLabels[i] + ",";
        output +=  eol + tab + tab + tab + inputLabels.back(); 
    }
    output += eol + tab + tab + "}," + eol + tab + tab + "{";

    auto argsList = opPtr->argsList();
    if (argsList.size() > 0)
    {
        for (std::size_t i = 0; i < argsList.size() - 1; ++i)
            output +=  eol + tab + tab + tab + "{ \"" + argsList[i] + "\", " + argsList[i] + " },";
        output +=  eol + tab + tab + tab + "{ \"" + argsList.back() + "\", " + argsList.back() + " }";
    }
    output += eol + tab + tab + "}," + eol + tab + tab + "name" + eol + tab + ");" + eol + "}";
    return output;

}

void mv::op::OpRegistry::generateCompositionAPI(const std::string& eol, const std::string& tab)
{

    std::ofstream incStream(utils::projectRootPath() + "/" + compAPIHeaderPath_, std::ios::out | std::ios::trunc);
    if (!incStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the header file during the CompositionAPI generation");

    incStream << "/*" << eol;
    incStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    incStream << "*/" << eol << eol;

    incStream << "#ifndef MV_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream << "#define MV_COMPOSITIONAL_MODEL_HPP_" << eol << eol; 
    incStream << "#include \"include/mcm/computation/model/op_model.hpp\"" << eol << eol;

    incStream << "namespace mv" << eol << eol;
    incStream << "{" << eol << eol;
    incStream << tab << "class CompositionalModel : private OpModel" << eol;
    incStream << tab << "{" << eol << eol;
    incStream << tab << "public:" << eol << eol;

    incStream << tab << tab << "CompositionalModel(OpModel& model);" << eol;
    incStream << tab << tab << "virtual ~CompositionalModel();" << eol << eol;

    auto opsList = getOpTypes({"exposed"});
    for (auto it = opsList.begin(); it != opsList.end(); ++it)
        incStream << tab << tab << "virtual " + getCompositionDecl_(*it) << ";" << eol;

    incStream << eol << tab << tab << "using OpModel::getSourceOp;" << eol;
    incStream << tab << tab << "using OpModel::addAttr;" << eol;
    incStream << tab << tab << "using OpModel::isValid;" << eol << eol;

    incStream << tab << "};" << eol << eol;
    incStream << "}" << eol << eol;

    incStream << "#endif //MV_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream.close();

    std::ofstream srcStream(utils::projectRootPath() + "/" + compAPISourcePath_, std::ios::out | std::ios::trunc);
    if (!srcStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the source file during the CompositionAPI generation");

    srcStream << "/*" << eol;
    srcStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    srcStream << "*/" << eol << eol;

    srcStream << "#include \"" << compAPIHeaderPath_ << "\"" << eol << eol;

    srcStream << "mv::CompositionalModel::CompositionalModel(OpModel& model) :" << eol;
    srcStream << tab << "OpModel(model)" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;

    srcStream << "mv::CompositionalModel::~CompositionalModel()" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;

    for (auto it = opsList.begin(); it != opsList.end(); ++it)
        srcStream << "mv::Data::TensorIterator mv::CompositionalModel::" + getCompositionDef_(*it, eol, tab) << eol << eol; 
    srcStream.close();

}

void mv::op::OpRegistry::generateRecordedCompositionAPI(const std::string& eol, const std::string& tab)
{

    std::ofstream incStream(utils::projectRootPath() + "/" + recordedCompAPIHeaderPath_, std::ios::out | std::ios::trunc);
    if (!incStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the header file during the RecordedCompositionAPI generation");

    incStream << "/*" << eol;
    incStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateRecordedCompositionAPI()" << eol;
    incStream << "*/" << eol << eol;

    incStream << "#ifndef MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream << "#define MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol << eol; 
    incStream << "#include \"" + compAPIHeaderPath_ + "\"" << eol << eol;

    incStream << "namespace mv" << eol << eol;
    incStream << "{" << eol << eol;
    incStream << tab << "class RecordedCompositionalModel : public CompositionalModel" << eol;
    incStream << tab << "{" << eol << eol;
    incStream << tab << "public:" << eol << eol;

    incStream << tab << tab << "RecordedCompositionalModel(CompositionalModel& model);" << eol;
    incStream << tab << tab << "virtual ~RecordedCompositionalModel();" << eol << eol;

    auto opsList = getOpTypes({"exposed"});
    for (auto it = opsList.begin(); it != opsList.end(); ++it)
        incStream << tab << tab << getCompositionDecl_(*it) << " override;" << eol;

    incStream << tab << "};" << eol << eol;
    incStream << "}" << eol << eol;

    incStream << "#endif //MV_RECORDED_COMPOSITIONAL_MODEL_HPP_" << eol;
    incStream.close();

    std::ofstream srcStream(utils::projectRootPath() + "/" + recordedCompAPISourcePath_, std::ios::out | std::ios::trunc);
    if (!srcStream.is_open())
        throw MasterError("OpRegistry", "Unable to create the source file during the CompositionAPI generation");

    srcStream << "/*" << eol;
    srcStream << tab << "DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()" << eol;
    srcStream << "*/" << eol << eol;

    srcStream << "#include \"" << recordedCompAPIHeaderPath_ << "\"" << eol << eol;

    srcStream << "mv::RecordedCompositionalModel::RecordedCompositionalModel(CompositionalModel& model) :" << eol;
    srcStream << tab << "CompositionalModel(model)" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;

    srcStream << "mv::RecordedCompositionalModel::~RecordedCompositionalModel()" << eol;
    srcStream << "{" << eol << eol;
    srcStream << "}" << eol << eol;

    for (auto it = opsList.begin(); it != opsList.end(); ++it)
    {

        srcStream << "mv::Data::TensorIterator mv::RecordedCompositionalModel::" << getCompositionDeclSig_(*it, true, false)
            << eol << "{" << eol;
        
        srcStream << tab << "Data::TensorIterator output = " << getCompositionCall_(*it) << ";" << eol;
        srcStream << getStringifiedOutputsCall_(*it, tab, eol);
        for (std::size_t i = 0; i < getOutputsCount(*it); ++i)
            srcStream << tab << "std::cout << output" + std::to_string(i) + " << std::endl;" << eol;

        auto argsStr = getStringifiedArgsCall_(*it);
        for (auto argIt = argsStr.begin(); argIt != argsStr.end(); ++argIt)
            srcStream << tab << "std::cout << " << *argIt << " << std::endl;" << eol;

        srcStream << getStringifiedInputsCall_(*it, tab, eol);
        for (std::size_t i = 0; i < getInputsCount(*it); ++i)
            srcStream << tab << "std::cout << input" + std::to_string(i) + " << std::endl;" << eol;

        srcStream << tab << "return output;" << eol << "}" << eol;
    }
    srcStream.close();

}