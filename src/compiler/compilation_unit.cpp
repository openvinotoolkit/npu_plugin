#include "include/mcm/compiler/compilation_unit.hpp"
#include "emu/manager.hpp"

const std::string mv::CompilationUnit::ma2480DefTargetDescPath_ = "/config/target/ma2480.json";
const std::string mv::CompilationUnit::ma2490DefTargetDescPath_ = "/config/target/release_kmb.json";
const std::string mv::CompilationUnit::compositionalModelRecordingsPath_ = "/recordings/";
const std::string mv::CompilationUnit::ma2480DefCompDescPath_ = "/config/compilation/release_ma2480.json";
const std::string mv::CompilationUnit::ma2490DefCompDescPath_ = "/config/compilation/release_kmb.json";
const std::string mv::CompilationUnit::ma2490EmulatorCompDescPath_ = "/contrib/mcm-emulator/config/compilation/emulator_kmb_SC-Prefetch1.json";

template <typename T1, typename T2>
std::vector<T1> read(const std::string& filepath)
{
    std::ifstream fileStream(filepath, std::ifstream::binary);
    if (!fileStream) return {};
    std::vector<T1> data;
    T2 aux;
    while (fileStream.read(&reinterpret_cast<char&>(aux), sizeof(aux))) data.emplace_back(aux);
    return data;
}

template <typename T1, typename T2>
void write(const std::vector<T1>& data, const std::string& filepath)
{
    std::ofstream file(filepath, std::ofstream::binary);
    T2 aux;
    for (const auto& value: data)
    {
        aux = value;
        file.write(&reinterpret_cast<char&>(aux), sizeof(aux));
    };
}

mv::CompilationUnit::CompilationUnit(const std::string& modelName) :
model_(new OpModel(modelName))
{
    MV_PROFILER_START;
}

mv::CompilationUnit::~CompilationUnit()
{
    MV_PROFILER_FINISH("profiler_output.prof");
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

    try 
    {   // parse recording on/off, but don't fail. Log message.
        if (compDescriptor_.getPassArg("initialize", "Singular", "GlobalConfigParams", "recorded_model"))
            model_->initRecordingFile("templateExampleNew.cpp");
    }
    catch (mv::AttributeError& e)
    {
        log(Logger::MessageType::Warning, "Could not find 'recorded_model' entry in 'GlobalConfigParams' section. Recording not enabled.");
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

bool mv::CompilationUnit::initialize()
{

    if (!passManager_.initialize(*model_, targetDescriptor_, compDescriptor_))
        return false;
    // Initialize resouces
    for (auto it = targetDescriptor_.memoryDefs().begin(); it != targetDescriptor_.memoryDefs().end(); ++it)
    {
        mv::DataModel dm(*model_);
        dm.addAllocator(it->first, it->second.size, it->second.alignment);
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

    // generate emulator results
    mv::Element globalParams = passList[0];
    bool emulator=false;
    if (globalParams.hasAttr("emulator_results") )
        emulator = globalParams.get<bool>("emulator_results");
    
    if (emulator)
        generateExpectedResults();

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

std::shared_ptr<std::vector<char>> mv::CompilationUnit::getBlob() const
{
    if(!completed())
        log(Logger::MessageType::Warning, "Getting a blob from compilation unit before completion");
    mv::RuntimeModel& rm = mv::RuntimeModel::getInstance(targetDescriptor_);
    return rm.getBlob();
}

/**
 * Generates a deep copy of the opmodel
 */
void mv::CompilationUnit::deepCopy(mv::OpModel& copyTo)
{
    // std::cout << "Original Model:" << std::endl;
    DataModel dm(*model_);
    // ControlModel cm(*model_);
    // std::cout << "OpsCount: " << model_->opsCount() << std::endl;
    // std::cout << "dataFlows: " << model_->dataFlowsCount() << std::endl;    
    // std::cout << "controlFlows: " << cm.controlFlowsCount() << std::endl;
    // std::cout << "tensorsCount: " << dm.tensorsCount() << std::endl;
    // std::cout << "populatedSize: " << dm.populatedTotalSize() << std::endl;
    // std::cout << "unpopulatedSize: " << dm.unpopulatedTotalSize() << std::endl;

    for(auto opIterator = model_->opBegin(); opIterator != model_->opEnd(); ++opIterator)
    {
        // getAttrs() returns map, defineOp() requires vector
        std::vector<std::pair<std::string, mv::Attribute>> vectAttrs;
        for (const auto &attr : opIterator->getAttrs())
            vectAttrs.push_back(attr);

        copyTo.defineOp(opIterator->getOpType(), opIterator->getInputTensor(), vectAttrs, opIterator->getName(), false, false);
    }

    // flow attribute is not being set for each operation in copied model. Manually adding...
    for (Data::FlowListIterator flow = dm.flowBegin(); flow != dm.flowEnd(); ++flow)
    {
        std::string tensorName = flow->getTensor()->getName();
        mv::Data::TensorIterator newTensor = copyTo.getTensor(tensorName);

        if(!newTensor->hasAttr("flows"))
        {
            std::set<std::string> toSet;
            newTensor->set<std::set<std::string>>("flows", toSet);
        }
        newTensor->get<std::set<std::string>>("flows").insert(flow->getName());
    }
}

void mv::CompilationUnit::generateExpectedResults()
{   
    log(mv::Logger::MessageType::Debug, "Initializing emulator...");
    std::cout << "Initializing emulator..." << std::endl;
    mv::CompilationUnit emUnit(model_->getName());
    mv::OpModel& emOM = emUnit.model();
    deepCopy(emOM);

    // std::cout << std::endl << "Copied:" << std::endl;
    // std::cout << "OpsCount: " << emOM.opsCount() << std::endl;
    // std::cout << "dataFlows: " << emOM.dataFlowsCount() << std::endl;
    // ControlModel cm1(emOM);
    // std::cout << "controlFlows: " << cm1.controlFlowsCount() << std::endl;
    // DataModel dm1(emOM);
    // std::cout << "tensorsCount: " << dm1.tensorsCount() << std::endl;
    // std::cout << "populatedSize: " << dm1.populatedTotalSize() << std::endl;
    // std::cout << "unpopulatedSize: " << dm1.unpopulatedTotalSize() << std::endl;

    emUnit.loadTargetDescriptor(mv::Target::ma2490);
    std::string emuCompPath = utils::projectRootPath() + ma2490EmulatorCompDescPath_;
    std::cout <<  "loading Comp desc: " << emuCompPath << std::endl;
    emUnit.loadCompilationDescriptor(emuCompPath);
    emUnit.compilationDescriptor().setPassArg("GlobalConfigParams", "emulator_results", false); // prevent infinite loop

    emUnit.initialize();
    emUnit.run();
    std::cout << "Compilation completed..." << std::endl;

    // initialize the Emulator Manager
    mv::emu::Manager emulatorManager(emOM);

    // set input tensor values
    std::vector<std::int64_t> input0Data = read<std::int64_t, std::uint8_t>("./input.dat");
    if (input0Data.empty() ) throw RuntimeError(*this, "Emulator required file 'input.dat' not found in current directory");

    mv::Data::OpListIterator opIt = emOM.getOps("Input")[0];
    mv::Data::TensorIterator tensorIt = emOM.getOps("Input")[0]->getOutputTensor()[0];
    
    emulatorManager.populate(*(emOM.getOps("Input")[0]->getOutputTensor()[0]), mv::Order::getZMajorID(4), input0Data);
    
    std::cout << "Generating results..." << std::endl;
    emulatorManager.run();
}