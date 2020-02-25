#include "include/graph_comparator/graph_comparator.hpp"

mv::tools::GraphComparator::GraphComparator() :
data1Buffer_(nullptr),
data2Buffer_(nullptr)
{

}

mv::tools::GraphComparator::~GraphComparator()
{

    if (data1Buffer_)
        delete [] data1Buffer_;

    if (data2Buffer_)
        delete [] data2Buffer_;

}

void mv::tools::GraphComparator::loadGraphFile_(const std::string& path, char *dataBuffer, MVCNN::GraphFileT& graph)
{

    if (!utils::fileExists(path))
        throw ArgumentError("tools:GraphComparator", "file:path", path, "File does not exist");

    inStream_.open(path, std::ios::binary | std::ios::in);
    inStream_.seekg(0, std::ios::end);
    std::size_t length = inStream_.tellg();

    if (length == 0)
        throw ArgumentError("tools:GraphComparator", "file:length", path, "File is empty");

    inStream_.seekg(0, std::ios::beg);

    if (dataBuffer != nullptr)
        delete dataBuffer;

    Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator", "Loading " + std::to_string(length) + " bytes "
        "from " + path);

    try
    {
        dataBuffer = new char[length];
    }   
    catch (const std::bad_alloc& e)
    {
        throw ArgumentError("tools:GraphComparator", "file:size", std::to_string(length), "Unable to allocate buffer");
    }

    inStream_.read(dataBuffer, length);
    inStream_.close();

    Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator", "Load successful");

    loadGraphFile_(dataBuffer, length, graph);

}

void mv::tools::GraphComparator::loadGraphFile_(const char *dataBuffer, std::size_t length, MVCNN::GraphFileT& graph)
{
    flatbuffers::Verifier verifier(reinterpret_cast<const unsigned char*>(dataBuffer), length);
    if (!MVCNN::VerifyGraphFileBuffer(verifier))
        throw ArgumentError("tools:GraphComparator", "file:content", "invalid", "GraphFile verification failed");
    Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator", "GraphFile verification successful");
    const MVCNN::GraphFile *graphPtr = MVCNN::GetGraphFile(dataBuffer);
    graphPtr->UnPackTo(&graph);
}

bool mv::tools::GraphComparator::compare(const MVCNN::GraphFileT& graph1, const MVCNN::GraphFileT& graph2)
{

    lastDiff_.clear();

    compare_(graph1, graph2, lastDiff_);

    if (data1Buffer_)
    {
        delete [] data1Buffer_;
        data1Buffer_ = nullptr;
    }
    if (data2Buffer_)
    {
        delete [] data2Buffer_;
        data2Buffer_ = nullptr;
    }

    return lastDiff_.empty();

}

bool mv::tools::GraphComparator::compare(const char *dataBuffer1, std::size_t length1, const char *dataBuffer2, std::size_t length2)
{

    MVCNN::GraphFileT graph1, graph2;
    loadGraphFile_(dataBuffer1, length1, graph1);
    loadGraphFile_(dataBuffer2, length2, graph2);

    return compare(graph1, graph2);

}


bool mv::tools::GraphComparator::compare(const std::string& path1, const std::string& path2)
{

    MVCNN::GraphFileT graph1, graph2;
    loadGraphFile_(path1, data1Buffer_, graph1);
    loadGraphFile_(path2, data2Buffer_, graph2);

    return compare(graph1, graph2);

}

MVCNN::GraphFileT mv::tools::GraphComparator::loadGraphFile(const std::string& path, char* dataBuffer)
{
    MVCNN::GraphFileT graph;
    loadGraphFile_(path, dataBuffer, graph);
    return graph;
}

const std::vector<std::string>& mv::tools::GraphComparator::lastDiff() const
{
    return lastDiff_;
}

void mv::tools::GraphComparator::compare_(const MVCNN::BarrierConfigurationTaskT& lhs, const MVCNN::BarrierConfigurationTaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    compare_(lhs.target, rhs.target, diff, label);
}

void mv::tools::GraphComparator::compare_(const MVCNN::BarrierReferenceT& lhs, const MVCNN::BarrierReferenceT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.update_barriers != rhs.update_barriers)
        diff.push_back(label + "::update_barriers");

    if (lhs.wait_barriers != rhs.wait_barriers)
        diff.push_back(label + "::wait_barriers");

}

void mv::tools::GraphComparator::compare_(const MVCNN::BarrierT& lhs, const MVCNN::BarrierT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.barrier_id != rhs.barrier_id)
        diff.push_back(label + "::barrier_id");

    if (lhs.consumer_count != rhs.consumer_count)
        diff.push_back(label + "::consumer_count");

    if (lhs.producer_count != rhs.producer_count)
        diff.push_back(label + "::producer_count");

}

void mv::tools::GraphComparator::compare_(const MVCNN::BinaryDataT& lhs, const MVCNN::BinaryDataT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    if (lhs.underlying_type != rhs.underlying_type)
        diff.push_back(label + "::underlying_type");
    if (lhs.length != rhs.length)
        diff.push_back(label + "::length");
    else
    {
        for(unsigned i = 0; i < lhs.length; ++i)
        {
            if(lhs.data[i] != rhs.data[i])
                diff.push_back(label + "::data::"+std::to_string(i));

        }
    }


}

void mv::tools::GraphComparator::compare_(const MVCNN::ControllerTaskT& lhs, const MVCNN::ControllerTaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.task.type != rhs.task.type)
        diff.push_back(label + "::barrier_id");
    {

        switch (lhs.task.type)
        {

            case MVCNN::ControllerSubTask_BarrierConfigurationTask:
                compare_(*lhs.task.AsBarrierConfigurationTask(), *rhs.task.AsBarrierConfigurationTask(),
                    diff, "BarrierConfigurationTask");
                break;
            
            case MVCNN::ControllerSubTask_TimerTask:
                compare_(*lhs.task.AsTimerTask(), *rhs.task.AsTimerTask(),
                    diff, "TimerTask");
                break;

            default:
                throw ArgumentError("tools:GraphComparator", "ControllerTaskT:task::type", "invalid", "Unexpected enum value");

        }

    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::Conv2DT& lhs, const MVCNN::Conv2DT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.bias, rhs.bias, diff, label + "::bias");

    if (lhs.dilation != rhs.dilation)
        diff.push_back(label + "::dilation");

    compare_(lhs.input, rhs.input, diff, label + "::input");
    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.padStyle != rhs.padStyle)
        diff.push_back(label + "::padStyle");

    if (lhs.padX != rhs.padX)
        diff.push_back(label + "::padX");

    if (lhs.padY != rhs.padY)
        diff.push_back(label + "::padY");

    if (lhs.radixX != rhs.radixX)
        diff.push_back(label + "::radixX");

    if (lhs.radixY != rhs.radixY)
        diff.push_back(label + "::radixY");

    if (lhs.strideX != rhs.strideX)
        diff.push_back(label + "::strideX");

    if (lhs.strideY != rhs.strideY)
        diff.push_back(label + "::strideY");

    compare_(lhs.weight, rhs.weight, diff, label + "::weight");

}

void mv::tools::GraphComparator::compare_(const MVCNN::CustomT& lhs, const MVCNN::CustomT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.data != rhs.data)
        diff.push_back(label + "::data");

    if (lhs.id != rhs.id)
        diff.push_back(label + "::id");

    if (lhs.length != rhs.length)
        diff.push_back(label + "::length");

}

void mv::tools::GraphComparator::compare_(const MVCNN::GraphFileT& lhs, const MVCNN::GraphFileT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    
    if (lhs.barrier_table.size() != rhs.barrier_table.size())
        diff.push_back(label + "barrier_table");
    else
    {
        for (std::size_t i = 0; i < lhs.barrier_table.size(); ++i)
            compare_(lhs.barrier_table.at(i), rhs.barrier_table.at(i), diff, label + "::barrier_table[" +
                std::to_string(i) + "]");
    }
    
    if (lhs.binary_data.size() != rhs.binary_data.size())
        diff.push_back(label + "binary_data");
    else
    {
        for (std::size_t i = 0; i < lhs.binary_data.size(); ++i)
            compare_(lhs.binary_data.at(i), rhs.binary_data.at(i), diff, label + "::binary_data[" +
                std::to_string(i) + "]");
    }
    
    compare_(lhs.header, rhs.header, diff, label + "::header");

    if (lhs.task_lists.size() != rhs.task_lists.size())
        diff.push_back(label + "task_lists");
    else
    {
        for (std::size_t i = 0; i < lhs.task_lists.size(); ++i)
            compare_(lhs.task_lists.at(i), rhs.task_lists.at(i), diff, label + "::task_lists[" +
                std::to_string(i) + "]");
    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::IndirectDataReferenceT& lhs, const MVCNN::IndirectDataReferenceT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.data_index != rhs.data_index)
        diff.push_back(label + "::data_index");

    if (lhs.sparsity_index != rhs.sparsity_index)
        diff.push_back(label + "::sparsity_index");

}

void mv::tools::GraphComparator::compare_(const MVCNN::GraphNodeT& lhs, const MVCNN::GraphNodeT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.name != rhs.name)
        diff.push_back(label + "::name");

    if (lhs.sinkID != rhs.sinkID)
        diff.push_back(label + "::sinkID");
    
    if (lhs.sourceID != rhs.sourceID)
        diff.push_back(label + "::sourceID");
    
    if (lhs.thisID != rhs.thisID)
        diff.push_back(label + "::thisID");

}

void mv::tools::GraphComparator::compare_(const MVCNN::MvTensorTaskT& lhs, const MVCNN::MvTensorTaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.layer.type != rhs.layer.type)
        diff.push_back(label + "::layer::type");
    else
    {

        switch (lhs.layer.type)
        {

            case MVCNN::SoftwareLayer_Conv2D:
                compare_(*lhs.layer.AsConv2D(), *rhs.layer.AsConv2D(),
                    diff, label + "::layer::Conv2D");
                break;
            
            case MVCNN::SoftwareLayer_Custom:
                compare_(*lhs.layer.AsCustom(), *rhs.layer.AsCustom(),
                    diff, label + "::layer::Custom");
                break;

            case MVCNN::SoftwareLayer_Passthrough:
                compare_(*lhs.layer.AsPassthrough(), *rhs.layer.AsPassthrough(),
                    diff, label + "::layer::Passthrough");
                break;

            case MVCNN::SoftwareLayer_Pooling:
                compare_(*lhs.layer.AsPooling(), *rhs.layer.AsPooling(),
                    diff, label + "::layer::Pooling");
                break;

            case MVCNN::SoftwareLayer_ReLU:
                compare_(*lhs.layer.AsReLU(), *rhs.layer.AsReLU(),
                    diff, label + "::layer::ReLU");
                break;

            default:
                throw ArgumentError("tools:GraphComparator", "MvTensorTaskT:layer::type", "invalid", "Unexpected enum value");

        }
        
    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::NCE1ConvT& lhs, const MVCNN::NCE1ConvT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.bias, rhs.bias, diff, label + "::bias");

    if (lhs.CMXSize != rhs.CMXSize)
        diff.push_back(label + "::CMXSize");

    if (lhs.concatOffset != rhs.concatOffset)
        diff.push_back(label + "::concatOffset");

    if (lhs.desc_count != rhs.desc_count)
        diff.push_back(label + "::desc_count");

    if (lhs.descriptors != rhs.descriptors)
        diff.push_back(label + "::descriptors");

    compare_(lhs.input, rhs.input, diff, label + "::input");

        if (lhs.inputSize != rhs.inputSize)
        diff.push_back(label + "::inputSize");

    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.outputSize != rhs.outputSize)
        diff.push_back(label + "::outputSize");

    if (lhs.overwriteInput != rhs.overwriteInput)
        diff.push_back(label + "::overwriteInput");

    if (lhs.reluSHVAcc != rhs.reluSHVAcc)
        diff.push_back(label + "::reluSHVAcc");

    if (lhs.shvNegSlope != rhs.shvNegSlope)
        diff.push_back(label + "::shvNegSlope");

    if (lhs.shvPosSlope != rhs.shvPosSlope)
        diff.push_back(label + "::shvPosSlope");

    if (lhs.streamingMask != rhs.streamingMask)
        diff.push_back(label + "::streamingMask");

    if (lhs.unloadCMX != rhs.unloadCMX)
        diff.push_back(label + "::unloadCMX");

    compare_(lhs.weight, rhs.weight, diff, label + "::weight");

}

void mv::tools::GraphComparator::compare_(const MVCNN::NCE1FCLT& lhs, const MVCNN::NCE1FCLT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.bias, rhs.bias, diff, label + "::bias");

    if (lhs.CMXSize != rhs.CMXSize)
        diff.push_back(label + "::CMXSize");

    if (lhs.concatOffset != rhs.concatOffset)
        diff.push_back(label + "::concatOffset");

    if (lhs.desc_count != rhs.desc_count)
        diff.push_back(label + "::desc_count");

    if (lhs.descriptors != rhs.descriptors)
        diff.push_back(label + "::descriptors");

    compare_(lhs.input, rhs.input, diff, label + "::input");

        if (lhs.inputSize != rhs.inputSize)
        diff.push_back(label + "::inputSize");

    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.outputSize != rhs.outputSize)
        diff.push_back(label + "::outputSize");

    if (lhs.overwriteInput != rhs.overwriteInput)
        diff.push_back(label + "::overwriteInput");

    if (lhs.reluSHVAcc != rhs.reluSHVAcc)
        diff.push_back(label + "::reluSHVAcc");

    if (lhs.shvNegSlope != rhs.shvNegSlope)
        diff.push_back(label + "::shvNegSlope");

    if (lhs.shvPosSlope != rhs.shvPosSlope)
        diff.push_back(label + "::shvPosSlope");

    if (lhs.streamingMask != rhs.streamingMask)
        diff.push_back(label + "::streamingMask");

    if (lhs.unloadCMX != rhs.unloadCMX)
        diff.push_back(label + "::unloadCMX");

    compare_(lhs.weight, rhs.weight, diff, label + "::weight");

}

void mv::tools::GraphComparator::compare_(const MVCNN::NCE1PoolT& lhs, const MVCNN::NCE1PoolT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.bias, rhs.bias, diff, label + "::bias");

    if (lhs.CMXSize != rhs.CMXSize)
        diff.push_back(label + "::CMXSize");

    if (lhs.concatOffset != rhs.concatOffset)
        diff.push_back(label + "::concatOffset");

    if (lhs.desc_count != rhs.desc_count)
        diff.push_back(label + "::desc_count");

    if (lhs.descriptors != rhs.descriptors)
        diff.push_back(label + "::descriptors");

    compare_(lhs.input, rhs.input, diff, label + "::input");

        if (lhs.inputSize != rhs.inputSize)
        diff.push_back(label + "::inputSize");

    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.outputSize != rhs.outputSize)
        diff.push_back(label + "::outputSize");

    if (lhs.overwriteInput != rhs.overwriteInput)
        diff.push_back(label + "::overwriteInput");

    if (lhs.reluSHVAcc != rhs.reluSHVAcc)
        diff.push_back(label + "::reluSHVAcc");

    if (lhs.shvNegSlope != rhs.shvNegSlope)
        diff.push_back(label + "::shvNegSlope");

    if (lhs.shvPosSlope != rhs.shvPosSlope)
        diff.push_back(label + "::shvPosSlope");

    if (lhs.streamingMask != rhs.streamingMask)
        diff.push_back(label + "::streamingMask");

    if (lhs.unloadCMX != rhs.unloadCMX)
        diff.push_back(label + "::unloadCMX");

    compare_(lhs.weight, rhs.weight, diff, label + "::weight");

}

void mv::tools::GraphComparator::compare_(const MVCNN::NCE1TaskT& lhs, const MVCNN::NCE1TaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.layer.type != rhs.layer.type)
        diff.push_back(label + "::layer::type");
    else
    {

        switch (lhs.layer.type)
        {

            case MVCNN::NCE1Layer_NCE1Conv:
                compare_(*lhs.layer.AsNCE1Conv(), *rhs.layer.AsNCE1Conv(),
                    diff, label + "::layer::NCE1Conv");
                break;
            
            case MVCNN::NCE1Layer_NCE1FCL:
                compare_(*lhs.layer.AsNCE1FCL(), *rhs.layer.AsNCE1FCL(),
                    diff, label + "::layer::NCE1FCL");
                break;

            case MVCNN::NCE1Layer_NCE1Pool:
                compare_(*lhs.layer.AsNCE1Pool(), *rhs.layer.AsNCE1Pool(),
                    diff, label + "::layer::NCE1Pool");
                break;

            default:
                throw ArgumentError("tools:GraphComparator", "NCE1TaskT::layer::type", "invalid", "Unexpected enum value");

        }

    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::NCE2TaskT& lhs, const MVCNN::NCE2TaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.invariant, rhs.invariant, diff, label + "::invariant");

    if (lhs.variant.size() != rhs.variant.size())
        diff.push_back(label + "::variant");
    else
    {
        for (std::size_t i = 0; i < lhs.variant.size(); ++i)
            compare_(lhs.variant.at(i), rhs.variant.at(i), diff, label + "::invariant[" + 
                std::to_string(i) + "]");
    }

    
}

void mv::tools::GraphComparator::compare_(const MVCNN::NCEInvariantFieldsT& lhs, const MVCNN::NCEInvariantFieldsT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.activation_window, rhs.activation_window, diff, label + "::activation_window");

    if (lhs.dpu_task_type != rhs.dpu_task_type)
        diff.push_back(label + "::dpu_task_type");

    compare_(lhs.input_data, rhs.input_data, diff, label + "::input_data");
    
    if (lhs.kernel_strideH != rhs.kernel_strideH)
        diff.push_back(label + "::kernel_strideH");

    if (lhs.kernel_strideW != rhs.kernel_strideW)
        diff.push_back(label + "::kernel_strideW");
    
    if (lhs.kernelH != rhs.kernelH)
        diff.push_back(label + "::kernelH");

    if (lhs.kernelW != rhs.kernelW)
        diff.push_back(label + "::kernelW");

    // if (lhs.nnshv_task.size() != rhs.nnshv_task.size())
    //     diff.push_back(label + "::nnshv_task");
    // else
    // {
    //     for (std::size_t i = 0; i < lhs.nnshv_task.size(); ++i)
    //         compare_(lhs.nnshv_task.at(i), rhs.nnshv_task.at(i), diff, label + "::nnshv_task[" + 
    //             std::to_string(i) + "]");
    // }
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::NCEVariantFieldsT& lhs, const MVCNN::NCEVariantFieldsT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.mpe_mode != rhs.mpe_mode)
        diff.push_back(label + "::mpe_mode");

    if (lhs.padBottom != rhs.padBottom)
        diff.push_back(label + "::clusterID");

    if (lhs.padLeft != rhs.padLeft)
        diff.push_back(label + "::padLeft");

    if (lhs.padRight != rhs.padRight)
        diff.push_back(label + "::padRight");

    if (lhs.padTop != rhs.padTop)
        diff.push_back(label + "::padTop");

    if (lhs.workload_end_X != rhs.workload_end_X)
        diff.push_back(label + "::workload_end_X");

    if (lhs.workload_end_Y != rhs.workload_end_Y)
        diff.push_back(label + "::workload_end_Y");

    if (lhs.workload_end_Z != rhs.workload_end_Z)
        diff.push_back(label + "::workload_end_Z");

    if (lhs.workload_start_X != rhs.workload_start_X)
        diff.push_back(label + "::workload_start_X");

    if (lhs.workload_start_Y != rhs.workload_start_Y)
        diff.push_back(label + "::workload_start_Y");

    if (lhs.workload_start_Z != rhs.workload_start_Z)
        diff.push_back(label + "::workload_start_Z");
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::NNDMATaskT& lhs, const MVCNN::NNDMATaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.compression != rhs.compression)
        diff.push_back(label + "::compression");

    compare_(lhs.src, rhs.src, diff, label + "::src");
    compare_(lhs.dst, rhs.dst, diff, label + "::dst");
}

void mv::tools::GraphComparator::compare_(const MVCNN::NNTensorTaskT& lhs, const MVCNN::NNTensorTaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.subtask.type != rhs.subtask.type)
        diff.push_back(label + "::subtask::type");
    else
    {

        switch (lhs.subtask.type)
        {

            case MVCNN::NNHelper_PPEAssist:
                compare_(*lhs.subtask.AsPPEAssist(), *rhs.subtask.AsPPEAssist(),
                    diff, label + "::subtask::PPEAssist");
                break;
            
            case MVCNN::NNHelper_PPEConfigure:
                compare_(*lhs.subtask.AsPPEConfigure(), *rhs.subtask.AsPPEConfigure(),
                    diff, label + "::subtask::PPEConfigure");
                break;
            
            default:
                throw ArgumentError("tools:GraphComparator", "NNTensorTaskT::subtask::type", "invalid", "Unexpected enum value");

        }

    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::PassthroughT& lhs, const MVCNN::PassthroughT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    
    compare_(lhs.input, rhs.input, diff, label + "::input");
    compare_(lhs.output, rhs.output, diff, label + "::input");

}

void mv::tools::GraphComparator::compare_(const MVCNN::PoolingT& lhs, const MVCNN::PoolingT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.dilation != rhs.dilation)
        diff.push_back(label + "::dilation");

    compare_(lhs.input, rhs.input, diff, label + "::input");
    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.padStyle != rhs.padStyle)
        diff.push_back(label + "::padStyle");

    if (lhs.padX != rhs.padX)
        diff.push_back(label + "::padX");

    if (lhs.padY != rhs.padY)
        diff.push_back(label + "::padY");

    if (lhs.radixX != rhs.radixX)
        diff.push_back(label + "::radixX");

    if (lhs.radixY != rhs.radixY)
        diff.push_back(label + "::radixY");

    if (lhs.strideX != rhs.strideX)
        diff.push_back(label + "::strideX");

    if (lhs.strideY != rhs.strideY)
        diff.push_back(label + "::strideY");

}

void mv::tools::GraphComparator::compare_(const MVCNN::PPEAssistT& lhs, const MVCNN::PPEAssistT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.op != rhs.op)
        diff.push_back(label + "::op");

}

void mv::tools::GraphComparator::compare_(const MVCNN::PPEConfigureT& lhs, const MVCNN::PPEConfigureT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.vals != rhs.vals)
        diff.push_back(label + "::vals");

}

void mv::tools::GraphComparator::compare_(const MVCNN::PPEFixedFunctionT& lhs, const MVCNN::PPEFixedFunctionT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.Clamp_High != rhs.Clamp_High)
        diff.push_back(label + "::Clamp_High");

    if (lhs.Clamp_Low != rhs.Clamp_Low)
        diff.push_back(label + "::Clamp_Low");

    if (lhs.Ops.size() != rhs.Ops.size())
        diff.push_back(label + "::Ops");
    else
    {
        for (std::size_t i = 0; i < lhs.Ops.size(); ++i)
            if (lhs.Ops.at(i) != rhs.Ops.at(i))
                diff.push_back(label + "::Ops[" + std::to_string(i) + "]");
    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::PPETaskT& lhs, const MVCNN::PPETaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.fixed_function, rhs.fixed_function, diff, label + "::fixed_function[");
    compare_(lhs.scale_data, rhs.scale_data, diff, label + "::scale_data");

}

void mv::tools::GraphComparator::compare_(const MVCNN::ReLUT& lhs, const MVCNN::ReLUT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.input, rhs.input, diff, label + "::input");

    if (lhs.opX != rhs.opX)
        diff.push_back(label + "::opX");

    compare_(lhs.output, rhs.output, diff, label + "::output");

    if (lhs.strideX != rhs.strideX)
        diff.push_back(label + "::strideX");

    if (lhs.strideY != rhs.strideY)
        diff.push_back(label + "::strideY");

}

void mv::tools::GraphComparator::compare_(const MVCNN::ResourcesT& lhs, const MVCNN::ResourcesT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.upa_shaves != rhs.upa_shaves)
        diff.push_back(label + "::upa_shaves");

    if (lhs.nce1_blocks != rhs.nce1_blocks)
        diff.push_back(label + "::nce1_blocks");

    if (lhs.nce2_blocks != rhs.nce2_blocks)
        diff.push_back(label + "::nce2_blocks");
    
    if (lhs.upa_shared_cmx != rhs.upa_shared_cmx)
        diff.push_back(label + "::upa_shared_cmx");

    if (lhs.nn_cmx_per_slice != rhs.nn_cmx_per_slice)
        diff.push_back(label + "::nn_cmx_per_slice");

    if (lhs.nn_cmx_slice_amount != rhs.nn_cmx_slice_amount)
        diff.push_back(label + "::nn_cmx_slice_amount");

    if (lhs.ddr_scratch != rhs.ddr_scratch)
        diff.push_back(label + "::ddr_scratch");
}

void mv::tools::GraphComparator::compare_(const MVCNN::SourceStructureT& lhs, const MVCNN::SourceStructureT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.first_ID != rhs.first_ID)
        diff.push_back(label + "::first_ID");

    if (lhs.nodes.size() != rhs.nodes.size())
        diff.push_back(label + "::links");
    else
    {
        for (std::size_t i = 0; i < lhs.nodes.size(); ++i)
            compare_(lhs.nodes.at(i), rhs.nodes.at(i), diff, label + "::links[" +
                std::to_string(i) + "]");
    }

}

void mv::tools::GraphComparator::compare_(const MVCNN::SummaryHeaderT& lhs, const MVCNN::SummaryHeaderT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    
    if (lhs.layer_count != rhs.layer_count)
        diff.push_back(label + "::layer_count");

    if (lhs.net_input.size() != rhs.net_input.size())
        diff.push_back(label + "::net_input");
    else
    {
        for (std::size_t i = 0; i < lhs.net_input.size(); ++i)
            compare_(lhs.net_input.at(i), rhs.net_input.at(i), diff, label + "::net_input[" + 
                std::to_string(i) + "]");
    }

    if (lhs.net_output.size() != rhs.net_output.size())
        diff.push_back(label + "::net_output");
    else
    {
        for (std::size_t i = 0; i < lhs.net_output.size(); ++i)
            compare_(lhs.net_output.at(i), rhs.net_output.at(i), diff, label + "::net_output[" + 
                std::to_string(i) + "]");
    }

    compare_(lhs.original_structure, rhs.original_structure, diff, label + "::original_structure");
    compare_(lhs.resources, rhs.resources, diff, label + "::resources");

    if (lhs.task_count != rhs.task_count)
        diff.push_back(label + "::task_count");

    compare_(lhs.version, rhs.version, diff, label + "::version");
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::TaskListT& lhs, const MVCNN::TaskListT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.content.size() != rhs.content.size())
        diff.push_back(label + "::content");
    else
    {
        for (std::size_t i = 0; i < lhs.content.size(); ++i)
            compare_(lhs.content.at(i), rhs.content.at(i), diff, label + "::net_input[" + 
                std::to_string(i) + "]");
    }
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::TaskT& lhs, const MVCNN::TaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.associated_barriers, rhs.associated_barriers, diff, label + "::associated_barriers");

    if (lhs.nodeID != rhs.nodeID)
        diff.push_back(label + "::nodeID");

    if (lhs.sourceTaskIDs != rhs.sourceTaskIDs)
        diff.push_back(label + "::sourceTaskIDs");

    if (lhs.task.type != rhs.task.type)
        diff.push_back(label + "::task::type");
    else
    {

        switch (lhs.task.type)
        {

            case MVCNN::SpecificTask_ControllerTask:
                compare_(*lhs.task.AsControllerTask(), *rhs.task.AsControllerTask(),
                    diff, label + "::task::ControllerTask");
                break;
            
            case MVCNN::SpecificTask_MvTensorTask:
                compare_(*lhs.task.AsMvTensorTask(), *rhs.task.AsMvTensorTask(),
                    diff, label + "::task::MvTensorTask");
                break;

            case MVCNN::SpecificTask_NCE1Task:
                compare_(*lhs.task.AsNCE1Task(), *rhs.task.AsNCE1Task(),
                    diff, label + "::task::NCE1Task");
                break;

            case MVCNN::SpecificTask_NCE2Task:
                compare_(*lhs.task.AsNCE2Task(), *rhs.task.AsNCE2Task(),
                    diff, label + "::task::NCE2Task");
                break;

            case MVCNN::SpecificTask_NNDMATask:
                compare_(*lhs.task.AsNNDMATask(), *rhs.task.AsNNDMATask(),
                    diff, label + "::task::NNDMATask");
                break;

            case MVCNN::SpecificTask_NNTensorTask:
                compare_(*lhs.task.AsNNTensorTask(), *rhs.task.AsNNTensorTask(),
                    diff, label + "::task::NNTensorTask");
                break;

            case MVCNN::SpecificTask_UPADMATask:
                compare_(*lhs.task.AsUPADMATask(), *rhs.task.AsUPADMATask(),
                    diff, label + "::task::UPADMATask");
                break;
                
            default:
                throw ArgumentError("tools:GraphComparator", "TaskT::task::type", "invalid", "Unexpected enum value");

        }

    }
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::TensorReferenceT& lhs, const MVCNN::TensorReferenceT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    compare_(lhs.data, rhs.data, diff, label + "::data");

    if (lhs.data_dtype != rhs.data_dtype)
        diff.push_back(label + "::data_dtype");

    if (lhs.dimensions != rhs.dimensions)
        diff.push_back(label + "::dimensions");

    if (lhs.leading_offset != rhs.leading_offset)
        diff.push_back(label + "::leading_offset");

    if (lhs.locale != rhs.locale)
        diff.push_back(label + "::locale");

    if (lhs.quant_scale != rhs.quant_scale)
        diff.push_back(label + "::quant_scale");

    if (lhs.quant_shift != rhs.quant_shift)
        diff.push_back(label + "::quant_shift");

    if (lhs.quant_zero != rhs.quant_zero)
        diff.push_back(label + "::quant_zero");

    if (lhs.strides != rhs.strides)
        diff.push_back(label + "::strides");

    if (lhs.trailing_offset != rhs.trailing_offset)
        diff.push_back(label + "::trailing_offset");
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::TensorT& lhs, const MVCNN::TensorT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.dataType != rhs.dataType)
        diff.push_back(label + "::dataType");

    if (lhs.dimX != rhs.dimX)
        diff.push_back(label + "::dimX");

    if (lhs.dimY != rhs.dimY)
        diff.push_back(label + "::dimY");

    if (lhs.dimZ != rhs.dimZ)
        diff.push_back(label + "::dimZ");

    if (lhs.location != rhs.location)
        diff.push_back(label + "::location");

    if (lhs.offset != rhs.offset)
        diff.push_back(label + "::offset");

    if (lhs.order != rhs.order)
        diff.push_back(label + "::order");

    if (lhs.strideX != rhs.strideX)
        diff.push_back(label + "::strideX");

    if (lhs.strideY != rhs.strideY)
        diff.push_back(label + "::strideY");

    if (lhs.strideZ != rhs.strideZ)
        diff.push_back(label + "::strideZ");
    
}

void mv::tools::GraphComparator::compare_(const MVCNN::TimerTaskT& lhs, const MVCNN::TimerTaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.id != rhs.id)
        diff.push_back(label + "::id");

    compare_(lhs.write_location, rhs.write_location, diff, label + "::write_location");

}

void mv::tools::GraphComparator::compare_(const MVCNN::UPADMATaskT& lhs, const MVCNN::UPADMATaskT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{
    
    compare_(lhs.dst, rhs.dst, diff, label + "::dst");
    compare_(lhs.src, rhs.src, diff, label + "::src");
}

void mv::tools::GraphComparator::compare_(const MVCNN::VersionT& lhs, const MVCNN::VersionT& rhs,
    std::vector<std::string>& diff, const std::string& label)
{

    if (lhs.hash != rhs.hash)
        diff.push_back(label + "::hash");

    if (lhs.majorV != rhs.majorV)
        diff.push_back(label + "::majorV");

    if (lhs.minorV != rhs.minorV)
        diff.push_back(label + "::minorV");

    if (lhs.patchV != rhs.patchV)
        diff.push_back(label + "::patchV");

}
