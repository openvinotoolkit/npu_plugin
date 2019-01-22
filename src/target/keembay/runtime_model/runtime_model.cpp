#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include "contrib/flatbuffers/include/flatbuffers/util.h"
#include "include/mcm/base/exception/argument_error.hpp"
#include <fstream>

mv::RuntimeModel::RuntimeModel()
    :graphFile_(MVCNN::GraphFileT())
{

}

mv::RuntimeModel::~RuntimeModel()
{

}

char * mv::RuntimeModel::serialize(int& bufferSize)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = MVCNN::CreateGraphFile(fbb, &graphFile_);
    MVCNN::FinishGraphFileBuffer(fbb, offset);
    bufferSize = fbb.GetSize();
    char * buffer = new char[bufferSize];
    std::memcpy(buffer, (char*)fbb.GetBufferPointer(), bufferSize);
    return buffer;
}

void mv::RuntimeModel::serialize(const std::string& path)
{
    int bufferSize;
    char * dataBuffer = serialize(bufferSize);
    flatbuffers::SaveFile(path.c_str(), dataBuffer, bufferSize, true);
    delete [] dataBuffer;
}

void mv::RuntimeModel::deserialize(const std::string& path)
{
    std::ifstream ifs(path.c_str(), std::ifstream::binary|std::ifstream::in);
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char * dataBuffer = new char[length];
    ifs.read(dataBuffer, length);
    ifs.close();
    deserialize(dataBuffer, length);
    delete [] dataBuffer;
}

void mv::RuntimeModel::deserialize(char * dataBuffer, int length)
{
    flatbuffers::Verifier verifier(reinterpret_cast<const unsigned char*>(dataBuffer), length);
    if (!MVCNN::VerifyGraphFileBuffer(verifier))
        throw ArgumentError("tools:GraphComparator", "file:content", "invalid", "GraphFile verification failed");
    Logger::log(mv::Logger::MessageType::Info, "RuntimeModel", "GraphFile verification successful");
    const MVCNN::GraphFile *graphPtr = MVCNN::GetGraphFile(dataBuffer);
    graphPtr->UnPackTo(&graphFile_);
}
