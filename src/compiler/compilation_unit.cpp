#include "include/mcm/compiler/compilation_unit.hpp"

const std::string mv::CompilationUnit::ma2480DefTargetDescPath_ = "/config/target/ma2480.json";
const std::string mv::CompilationUnit::ma2490DefTargetDescPath_ = "/config/target/ma2490.json";
const std::string mv::CompilationUnit::compositionalModelRecordingsPath_ = "/recordings/";
const std::string mv::CompilationUnit::ma2480DefCompDescPath_ = "/config/compilation/release_ma2480.json";
const std::string mv::CompilationUnit::ma2490DefCompDescPath_ = "/config/compilation/release_ma2490.json";

mv::CompilationUnit::CompilationUnit(const std::string& modelName) :
model_(new OpModel(modelName)),
recordedModel_(new RecordedCompositionalModel(*model_, compositionalModelRecordingsPath_))
{
    MV_PROFILER_START;
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
    MV_PROFILER_FINISH("profiler_output.prof");
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

bool mv::CompilationUnit::loadCompilationDescriptor(Target target)
{
    std::string descPath;

    switch (target)
    {
        case Target::ma2480:
        {
            descPath = utils::projectRootPath() + ma2480DefCompDescPath_;
            break;
        }
        case Target::ma2490:
        {
            descPath = utils::projectRootPath() + ma2490DefCompDescPath_;
            break;
        }
        default:
            return false;
    }

    return loadCompilationDescriptor(descPath);
}

bool mv::CompilationUnit::loadTargetDescriptor(Target target)
{

    switch (target)
    {

        case Target::ma2480:
        {
            std::string descPath = utils::projectRootPath() + ma2480DefTargetDescPath_;
            return loadTargetDescriptor(descPath);
        }

        case Target::ma2490:
        {
            std::string descPath = utils::projectRootPath() + ma2490DefTargetDescPath_;
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

mv::Element mv::CompilationUnit::runStep()
{
    return passManager_.step();
}

mv::Element mv::CompilationUnit::run()
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PHASE);
    Element output("CompilationOutput");
    output.set<std::string>("ModelName", model_->getName());
    std::vector<mv::Element> passList = compDescriptor_.serializePassList();
    passManager_.loadPassList(passList);

    while (!passManager_.completed())
    {
        MV_PROFILE_VIRTUAL_MEM;
        MV_PROFILE_PHYSICAL_MEM;
        #ifdef MV_PROFILER_ENABLED
        std::size_t ops = model_->opsCount();
        std::size_t dataFlows = model_->dataFlowsCount();
        ControlModel cm(*model_);
        std::size_t controlFlows = cm.controlFlowsCount();
        DataModel dm(*model_);
        std::size_t tensors = dm.tensorsCount();
        std::size_t populatedSize = dm.populatedTotalSize();
        std::size_t unpopulatedSize = dm.unpopulatedTotalSize();
        MV_PROFILED_VARIABLE(ops, MV_PROFILER_COLOR_GREEN);
        MV_PROFILED_VARIABLE(dataFlows, MV_PROFILER_COLOR_ROSE);
        MV_PROFILED_VARIABLE(controlFlows, MV_PROFILER_COLOR_RED);
        MV_PROFILED_VARIABLE(tensors, MV_PROFILER_COLOR_BLUE);
        MV_PROFILED_VARIABLE(populatedSize, MV_PROFILER_COLOR_ORANGE);
        MV_PROFILED_VARIABLE(unpopulatedSize, MV_PROFILER_COLOR_LIME);
        #endif

        output = passManager_.step();
    }

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

mv::BlobBinary mv::CompilationUnit::getBlob() const
{
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance();
    return rm.getBlob();
}
