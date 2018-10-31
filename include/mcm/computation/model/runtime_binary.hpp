#ifndef RUNTIME_BINARY_HPP_
#define RUNTIME_BINARY_HPP_

#include <string>

namespace mv
{

    class RuntimeBinary 
    {
    
    protected:
        char* data;
        uint32_t fileSize;
        uint32_t bufferSize;
        bool fileEnabled;
        bool RAMEnabled;
        std::string fileName;
        std::string binaryName;
        int bytePointer ;
        
    public:

        RuntimeBinary();
        ~RuntimeBinary();
        int getFileSize();
        int getBufferSize();
        bool getRAMEnabled();
        bool getFileEnabled();
        bool setRAMEnabled(bool flag);
        bool setFileEnabled(bool flag);
        std::string getBinaryName();
        std::string getFileName();
        bool setFileName(std::string newName);
        bool getBuffer(std::string newName, uint32_t newSize);
        bool getBuffer(uint32_t newSize);
        bool writeBuffer(char sourceBuf[4096], int numBytes);
        bool dumpBuffer(std::string testFileName);

    };
}

#endif // RUNTIME_BINARY_HPP_
