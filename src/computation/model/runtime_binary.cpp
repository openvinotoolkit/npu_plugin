#include "include/mcm/computation/model/runtime_binary.hpp"
#include <iostream>

mv::RuntimeBinary::RuntimeBinary() :
binaryName("NULL"),
fileName("test_RTB_00.blob"),
fileSize(0),
bufferSize(0),
RAMEnabled(true),
fileEnabled(true),
data(nullptr)
{
}

mv::RuntimeBinary::~RuntimeBinary()
{
    if(data)
    {
        delete [] data;
    }
}

bool mv::RuntimeBinary::RuntimeBinary::getBuffer(std::string newName, int newSize)
{
    data = new char[newSize+RAMAlign];
    bufferSize = newSize;
    binaryName = newName;
    // just for testing, fill RAM buffer with background pattern
    for (int i=0; i<newSize; i=i+4)
    {
        data[i]=1;
        data[i+1]=2;
        data[i+2]=3;
        data[i+3]=4;
    }
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::getBuffer(int newSize)
{  
    data = new char[newSize+RAMAlign];
    bufferSize = newSize;
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::writeBuffer(char sourceBuf[4096], int numBytes)
{
    for (int i=0; i<numBytes; i++)
    {
        data[fileSize+i] = sourceBuf[i];
    }
    fileSize += numBytes ;
    return true ;
}

std::string mv::RuntimeBinary::RuntimeBinary::getBinaryName()
{
    return binaryName ;
}

std::string mv::RuntimeBinary::RuntimeBinary::getFileName()
{
    return fileName ;
}

bool mv::RuntimeBinary::RuntimeBinary::setFileName(std::string newName)
{
    fileName = newName;
    return true ;
}

int mv::RuntimeBinary::RuntimeBinary::getFileSize()
{
    return fileSize ;
}

int mv::RuntimeBinary::RuntimeBinary::getBufferSize()
{
    return bufferSize ;
}

bool mv::RuntimeBinary::RuntimeBinary::getRAMEnabled()
{
    return RAMEnabled ;
}

bool mv::RuntimeBinary::RuntimeBinary::setRAMEnabled(bool flag)
{
    RAMEnabled = flag ;
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::getFileEnabled()
{
    return fileEnabled ;
}

bool mv::RuntimeBinary::RuntimeBinary::setFileEnabled(bool flag)
{
    fileEnabled = flag ; 
    return true ;
}

bool mv::RuntimeBinary::RuntimeBinary::dumpBuffer(std::string testFileName)
{
    std::cout << " Dumping Blob RAM Buffer to "<< testFileName << " :bufferSize, fileSize = "<< bufferSize << " " << fileSize << std::endl;
    FILE *testfp ;

    if ((testfp = fopen(testFileName.c_str(), "wb")) == NULL)
    {
        std::cout << "ERROR: Could not open output file"<< testFileName << std::endl;
    }

    fwrite(data,1,fileSize,testfp);
    fclose(testfp);

    return true ;
}

