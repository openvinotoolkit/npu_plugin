#include "include/mcm/compiler/compilation_unit.hpp"

const std::string mv::CompilationUnit::ma2480DefDescPath_ = "/config/target/ma2480.json";
const std::string mv::CompilationUnit::compositionalModelRecordingsPath_ = "/recordings/";

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

    JSONTextParser parser(jsonParserBufferLenght_);

    try
    {

        json::Value jsonRoot;
        if (!parser.parseFile(filePath, jsonRoot))
        {
            throw ArgumentError(*this, "filePath", filePath,
                "Unable to parse compilation descriptor - error reading");
        }
        if (jsonRoot.valueType() != json::JSONType::Object)
            return false;
        else
            compilationDescriptor_ = jsonRoot.get<json::Object>();

    }
    catch (ParsingError& e)
    {
        return false;
    }
    return true;

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

        default:
            return false;

    }

    return false;

}

mv::PassManager& mv::CompilationUnit::passManager()
{
    return passManager_;
}

mv::json::Object& mv::CompilationUnit::compilationDescriptor()
{
    return compilationDescriptor_;
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

    if (!passManager_.initialize(*model_, targetDescriptor_, compilationDescriptor_))
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
