#ifndef RUNTIME_BINARY_HPP_
#define RUNTIME_BINARY_HPP_

#include <string>

namespace mv
{

    class RuntimeBinary 
    {
    
    protected:
        char* data;
        int size;
        std::string name;
        
    public:

        RuntimeBinary(std::string nameOfBinary, int sizeOfBuffer);
        ~RuntimeBinary();
        bool getBuffer(int sizeOfBuffer);
        int getSize();
        std::string getName();
        bool getBuffer(std::string newName, int newSize);

    };
}

#endif // RUNTIME_BINARY_HPP_
