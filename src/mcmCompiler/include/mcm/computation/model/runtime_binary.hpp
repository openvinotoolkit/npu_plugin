#ifndef RUNTIME_BINARY_HPP_
#define RUNTIME_BINARY_HPP_

#include <string>

namespace mv
{

    class RuntimeBinary 
    {
    
    private:
        char* data_;
        std::size_t fileSize_;
        std::size_t bufferSize_;
        bool fileEnabled_;
        bool RAMEnabled_;
        std::string fileName_;
        std::string binaryName_;
        
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
        bool setBinaryName(std::string newName);
        bool getBuffer(std::string newName, std::size_t newSize);
        bool getBuffer(std::size_t newSize);
        char* getDataPointer();
        bool writeBuffer(char *sourceBuf, int numBytes);
        bool dumpBuffer(std::string testFileName);

    };
}

#endif // RUNTIME_BINARY_HPP_
