#include "include/mcm/compiler/compilation_unit.hpp"

const std::string mv::CompilationUnit::ma2480DefDescPath_ = "/config/target/ma2480.json";
const std::string mv::CompilationUnit::ma2490DefDescPath_ = "/config/target/ma2490.json";
const std::string mv::CompilationUnit::compositionalModelRecordingsPath_ = "/recordings/";
const std::string mv::CompilationUnit::compilationDescPath_ = "/config/compilation/default_ma2480.json";

mv::CompilationUnit::CompilationUnit(const std::string& modelName) :
model_(new OpModel(modelName)),
recordedModel_(new RecordedCompositionalModel(*model_, compositionalModelRecordingsPath_))
{

}

/*void mv::CompilationUnit::loadModelFromJson(const std::string &path)
{

    mv::JSONTextParser parser;
    mv::json::Value value;
    parser.parseFile(path, value);
    delete model_;
    model_ = new OpModel(value);
    if(mv::Jsonable::constructBoolTypeFromJson(value["has_populated_tensors"]))
    {
        size_t lastindex = path.find_last_of(".");
        std::string pathNoExt(path.substr(0, lastindex));

        for(auto tensorIt = model_->tensorBegin(); tensorIt != model_->tensorEnd(); ++tensorIt)
        {
            if(!tensorIt->isPopulated())
                continue;
            std::string currentTensorInputPath(pathNoExt+"_"+tensorIt->getName());
            std::ifstream currentTensorInputStream(currentTensorInputPath, std::ios::in | std::ios::binary);
            std::vector<double> tensorData(tensorIt->getShape().totalSize());
            currentTensorInputStream.read(reinterpret_cast<char*>(&tensorData[0]), tensorData.size() * sizeof(tensorData[0]));
            tensorIt->populate(tensorData, tensorIt->getOrder());
            currentTensorInputStream.close();
        }
    }

}*/

mv::CompilationUnit::~CompilationUnit()
{
	delete model_;
    delete recordedModel_;
}

bool mv::CompilationUnit::loadTargetDescriptor(const std::string& path)
{
    try
    {
        return targetDescriptor_.load(path);
    }
    catch (ArgumentError& e)
    {
        log(Logger::MessageType::Error, e.what());
        return false;
    }

    return true;

}

bool mv::CompilationUnit::loadCompilationDescriptor(const std::string& filePath)
{
    try
    {
        mv::json::Object jsonDesc = mv::CompilationDescriptor::load(filePath);
        compDescriptor_ = CompilationDescriptor(jsonDesc);
    }
    catch (ParsingError& e)
    {
        return false;
    }

    return true;
}

bool mv::CompilationUnit::loadDefaultCompilationDescriptor()
{
    std::string filePath = utils::projectRootPath() + compilationDescPath_;
    return loadCompilationDescriptor(filePath);
}

bool mv::CompilationUnit::loadTargetDescriptor(Target target)
{

    switch (target)
    {

        case Target::ma2480:
        {
            std::string descPath = utils::projectRootPath() + ma2480DefDescPath_;
            return loadTargetDescriptor(descPath);
        }

        case Target::ma2490:
        {
            std::string descPath = utils::projectRootPath() + ma2490DefDescPath_;
            return loadTargetDescriptor(descPath);
        }

        default:
            return false;

    }

    return false;

}

mv::CompilationDescriptor& mv::CompilationUnit::compilationDescriptor()
{
    return compDescriptor_;
}

mv::OpModel& mv::CompilationUnit::model()
{
    return *model_;
}

mv::CompositionalModel& mv::CompilationUnit::recordedModel()
{
    return *recordedModel_;
}

bool mv::CompilationUnit::initialize()
{

    if (!passManager_.initialize(*model_, targetDescriptor_, compDescriptor_))
        return false;
    // Initialize resouces
    for (auto it = targetDescriptor_.memoryDefs().begin(); it != targetDescriptor_.memoryDefs().end(); ++it)
    {
        mv::DataModel dm(*model_);
        dm.addAllocator(it->first, it->second.size, it->second.alignment, it->second.dataTypeSize);
    }

    return true;

}

mv::json::Object mv::CompilationUnit::runStep()
{
    return passManager_.step();
}

mv::json::Object mv::CompilationUnit::run()
{
    json::Object output;
    std::vector<mv::Element> passList = compDescriptor_.serializePassList();
    passManager_.loadPassList(passList);

    while (!passManager_.completed())
        output = passManager_.step();
    return output;
}

bool mv::CompilationUnit::completed() const
{
    return passManager_.completed();
}

std::string mv::CompilationUnit::getLogID() const
{
    return "CompilationUnit";
}
