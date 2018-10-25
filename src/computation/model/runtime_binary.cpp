#include "include/mcm/computation/model/runtime_binary.hpp"

mv::RuntimeBinary::RuntimeBinary(std::string nameOfBinary, int sizeOfBinary) :
name(nameOfBinary),
size(sizeOfBinary)
{
}

int mv::RuntimeBinary::RuntimeBinary::getSize()
{
   return size ;
}

bool mv::RuntimeBinary::RuntimeBinary::getBuffer(std::string newName, int newSize)
{
   data = new char[newSize];
   for( int i = 1; i < newSize-10; i = i + 1 ) {
       data[i] = i ;
   }
   size = newSize;
   return true ;
}

std::string mv::RuntimeBinary::RuntimeBinary::getName()
{
   return name ;
}

