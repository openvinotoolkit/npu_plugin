#include "include/mcm/compiler/compilation_unit.hpp"

const std::string mv::CompilationUnit::ma2480DefDescPath_ = "/config/target/ma2480.json";

mv::Logger& mv::CompilationUnit::logger_ = mv::ComputationModel::logger();

mv::CompilationUnit::CompilationUnit(mv::Logger::VerboseLevel verboseLevel, bool logTime) :
model_(new OpModel(verboseLevel, logTime))
{

}

void mv::CompilationUnit::loadModelFromJson(const std::string &path)
{
    mv::JSONTextParser parser;
    mv::json::Value value;
    parser.parseFile(path, value);
    OpModel * oldModel = model_;
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
            mv::dynamic_vector<float> tensorData(tensorIt->getShape().totalSize());
            currentTensorInputStream.read(reinterpret_cast<char*>(&tensorData[0]), tensorData.size() * sizeof(tensorData[0]));
            tensorIt->populate(tensorData, tensorIt->getOrder());
            currentTensorInputStream.close();
        }
    }
    delete oldModel;
}

mv::CompilationUnit::~CompilationUnit()
{
    delete model_;
}

bool mv::CompilationUnit::loadTargetDescriptor(const std::string& path)
{
    try
    {
        return targetDescriptor_.load(path);
    }
    catch (ArgumentError& e)
    {
        logger_.log(Logger::MessageType::MessageError, e.what());
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

mv::CompositionalModel& mv::CompilationUnit::model()
{
    return *model_;
}

bool mv::CompilationUnit::initialize()
{
    return passManager_.initialize(*model_, targetDescriptor_, compilationDescriptor_);
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