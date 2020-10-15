#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

// Generic 4KB output buffer supporting bit-level output to file.
// Buffer empties at 3800 level. Assumes add size < 296 to prevent
// excessive size checking during adds.

namespace mv
{

        WBuffer::WBuffer() :
        bitPointer(0),
        fileSize(0)
        {
        }

        std::size_t WBuffer::getBitPointer()
        {
            return this->bitPointer;
        }

        uint64_t WBuffer::getFileSize()
        {
            return this->fileSize;
        }

        void WBuffer::open(std::shared_ptr<mv::RuntimeBinary> rtBin )
         {
             bp = rtBin ; 
             if (bp->getFileEnabled())
             {
                 outputFile.open(bp->getFileName(), std::ios::out | std::ios::binary);
                 if (!(outputFile.is_open()))
                 {
                     throw mv::ArgumentError("file_buffer", "output", bp->getFileName(), "Unable to open output file");
                 }
             }
         }

         uint64_t WBuffer::End()
         {
             int j ;
             int bytes ;

             if (bitPointer>0)
             {
                 bytes = (bitPointer / 8);
                 j = (bitPointer % 8) ;
                 if ((j % 8) != 0 )
                 {
                     Data[bytes] = Data[bytes]<<(8-j);
                     bytes++;
                 }
                 if (bp->getFileEnabled())
                 {   
                    for (int i=0; i<bytes; i++)
                    {   
                        outputFile << Data[i];
                    }
                 }
                 if (bp->getRAMEnabled())
                 {   
                     bp->writeBuffer(Data, bytes);
                 }
                 fileSize+=bytes;
             }
   
             if (bp->getFileEnabled())
             {
                 outputFile.close();
             }
             return (fileSize);
         }

}  // end namespace mv
