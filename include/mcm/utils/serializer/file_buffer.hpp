#ifndef WBUFFER_HPP_
#define WBUFFER_HPP_

#include <iostream>
#include <fstream>
#include <memory>
#include "include/mcm/computation/model/runtime_binary.hpp"

// Generic 4KB output buffer supporting bit-level output to file.
// Buffer empties at 3800 level. Assumes add size < 296 to prevent
// excessive size checking during adds.
namespace mv
{
class WBuffer
{
    private:
        static const int wbuffer_size = 4096 ;
        static const int wlevel = 3800*8 ;      //buuffer empty level in bits
        char Data[wbuffer_size] ;
        std::size_t bitPointer ;
        std::ofstream outputFile ;
        std::shared_ptr<mv::RuntimeBinary> bp ; 
        uint64_t fileSize ;

    public:
        WBuffer();
        std::size_t getBitPointer();
        uint64_t getFileSize();

        template <typename number_T>
        number_T align8(number_T number_2_round)
        {
            number_T align_size = 8 ;
            number_T retval = (number_2_round/align_size)*align_size ;
            if ((number_2_round % align_size) != 0)
            {
                retval += align_size ;
            }
            return retval ;
        }

        template <typename number_T, typename size_T>
        number_T align(number_T number_2_round, size_T align_size)
        {
            return align_size+(number_2_round/align_size)*align_size ;
        }

        template <typename field_T>
        void AddBytes(int numbytes, field_T field)
        {
            int byte_pointer ;
            int j;

            byte_pointer = (bitPointer / 8);

            // dump buffer if full
            if ((numbytes*8+bitPointer) > wlevel)
            {
                if (bp->getFileEnabled())
                {
                    for (int i=0; i<byte_pointer; i++)
                    {
                        outputFile << Data[i];
                    }
                }
                if (bp->getRAMEnabled())
                {
                    bp->writeBuffer(Data, byte_pointer);
                }
                fileSize+=byte_pointer;
                bitPointer = bitPointer - (8*byte_pointer);
                Data[0]=Data[byte_pointer] ;
                Data[1]=Data[byte_pointer+1] ;
                byte_pointer = bitPointer / 8;
            }

            // write numbytes bytes to output buffer
            for (j=0; j<numbytes; j++)
            {
                Data[byte_pointer+j] = (field >> 8*j )  & 0xff;
            }

            bitPointer+=8*numbytes;

         }

        template <typename field_T>
        void AddBits(int numbits, field_T field)
        {
            int bytes ;
            int j;
            char thisbit ;

            bytes = (bitPointer / 8);

            // field needs to be of type that supports bit level manipulation
            uint32_t index = *reinterpret_cast<uint32_t*>(&field);

            // dump buffer if full
            if ((numbits+bitPointer) > wlevel)

            {
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
                bitPointer = bitPointer - (8*bytes);
                Data[0]=Data[bytes] ;
                Data[1]=Data[bytes+1] ;
                bytes = bitPointer / 8;
            }

            // write numbits bits to output buffer
            for (j=(numbits-1); j>=0; j--)
            {
                thisbit = ((index>>j) & (0x01)) ;
                Data[bytes]=Data[bytes]<<1;
                Data[bytes]=Data[bytes] | thisbit ;
                bitPointer++;
                if ((bitPointer % 8) == 0)
                {
                    bytes++;
                }
             }
         }

         void open(std::shared_ptr<mv::RuntimeBinary> rtBin );

         uint64_t End();

};   // end WBuffer class
} // end namespace mv
#endif // WBUFFER_HPP_
