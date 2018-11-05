#include "include/mcm/computation/model/runtime_binary.hpp"
#include <iostream>

mv::RuntimeBinary::RuntimeBinary() :
binaryName_("NULL"),
fileName_("mcmCompile.blob"),
fileSize_(0),
bufferSize_(0),
RAMEnabled_(true),
fileEnabled_(true),
data_(nullptr)
{
}

mv::RuntimeBinary::~RuntimeBinary()
{
    if(data_)
    {
        delete [] data_;
    }
}

bool mv::RuntimeBinary::RuntimeBinary::getBuffer(std::string newName, std::size_t newSize)
{
//    bufferSize_ = newSize;
//    data_ = new char[bufferSize_];
    binaryName_ = newName;
    if (getBuffer(newSize))
    {
        return true ;
    }
    return false;
}

bool mv::RuntimeBinary::RuntimeBinary::getBuffer(std::size_t newSize)
{  
    bufferSize_ = newSize;
    if (data_ )
    {
        std::cout << "WARNING: RuntimeBinary already exists. Destroying previous allocation." << std::endl;
        delete [] data_;
    }
    data_ = new char[bufferSize_];
    return true ;
}

char* mv::RuntimeBinary::RuntimeBinary::getDataPointer()
{
    return data_ ;
}

bool mv::RuntimeBinary::RuntimeBinary::writeBuffer(char* sourceBuf, int numBytes)
{
    for (int i=0; i<numBytes; i++)
    {
        data_[fileSize_+i] = sourceBuf[i];
    }
    fileSize_ += numBytes ;
    return true ;
}

std::string mv::RuntimeBinary::RuntimeBinary::getBinaryName()
{
    return binaryName_ ;
}

std::string mv::RuntimeBinary::RuntimeBinary::getFileName()
{
    return fileName_ ;
}

bool mv::RuntimeBinary::RuntimeBinary::setFileName(std::string newName)
{
    fileName_ = newName;
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::setBinaryName(std::string newName)
{
    binaryName_ = newName;
    return true ;
}

int mv::RuntimeBinary::RuntimeBinary::getFileSize()
{
    return fileSize_ ;
}

int mv::RuntimeBinary::RuntimeBinary::getBufferSize()
{
    return bufferSize_ ;
}

bool mv::RuntimeBinary::RuntimeBinary::getRAMEnabled()
{
    return RAMEnabled_ ;
}

bool mv::RuntimeBinary::RuntimeBinary::setRAMEnabled(bool flag)
{
    RAMEnabled_ = flag ;
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::getFileEnabled()
{
    return fileEnabled_ ;
}

bool mv::RuntimeBinary::RuntimeBinary::setFileEnabled(bool flag)
{
    fileEnabled_ = flag ; 
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::dumpBuffer(std::string testFileName)
{
    std::cout << " Dumping Blob RAM Buffer to "<< testFileName << " :bufferSize_, fileSize_ = "<< bufferSize_ << " " << fileSize_ << std::endl;
    FILE *testfp ;

    if ((testfp = fopen(testFileName.c_str(), "wb")) == NULL)
    {
        std::cout << "ERROR: Could not open output file"<< testFileName << std::endl;
    }

    fwrite(data_,1,fileSize_,testfp);
    fclose(testfp);

    return true ;
}

