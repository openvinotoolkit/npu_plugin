
#ifndef WBUFFER_HPP_
#define WBUFFER_HPP_

#include <iostream>

// Generic 4KB output buffer supporting bit-level output to file.
// Buffer empties at 3800 level. Assumes add size < 296 to prevent
// excessive size checking during adds.

class WBuffer
{
    private:
        static const int wbuffer_size = 4096 ;
        static const int wlevel = 3800 ;
        char Data[wbuffer_size] ;
        int BitPointer ;
        FILE *fp ;

    public:
        uint64_t FileSize ;
        WBuffer()
        {
            BitPointer  = 0 ;
            FileSize = 0 ;
        }

        int getPointer(){
            return this->BitPointer;
        }

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

            byte_pointer = (BitPointer / 8);

            // dump buffer if full
            if ((numbytes*8+BitPointer) > wlevel)
            {
                fwrite(&Data,1,byte_pointer,fp);
                FileSize+=byte_pointer;
                BitPointer = BitPointer - (8*byte_pointer);
                Data[0]=Data[byte_pointer] ;
                Data[1]=Data[byte_pointer+1] ;
                byte_pointer = BitPointer / 8;
            }

            // write numbytes bytes to output buffer
            for (j=0; j<numbytes; j++)
            {
                Data[byte_pointer+j] = (field >> 8*j )  & 0xff;
            }

            BitPointer+=8*numbytes;

         }

        template <typename field_T>
        void AddBits(int numbits, field_T field)
        {
            int bytes ;
            int j;
            char thisbit ;

            bytes = (BitPointer / 8);

            // field needs to be of type that supports bit level manipulation
            uint32_t index = *reinterpret_cast<uint32_t*>(&field);

            // dump buffer if full
            if ((numbits+BitPointer) > wlevel)
            {
                fwrite(&Data,1,bytes,fp);
                FileSize+=bytes;
                BitPointer = BitPointer - (8*bytes);
                Data[0]=Data[bytes] ;
                Data[1]=Data[bytes+1] ;
                bytes = BitPointer / 8;
            }

            // write numbits bits to output buffer
            for (j=(numbits-1); j>=0; j--)
            {
                thisbit = ((index>>j) & (0x01)) ;
                Data[bytes]=Data[bytes]<<1;
                Data[bytes]=Data[bytes] | thisbit ;
                BitPointer++;
                if ((BitPointer % 8) == 0)
                {
                    bytes++;
                }
             }
         }

         void open(char const *out_file_name)
         {
            if ((fp = fopen(out_file_name, "w")) == NULL)
             {
                 std::cout << "ERROR: Could not open output file" << std::endl;
             }
         }

         uint64_t End()
         {
             int j ;
             int bytes ;

             if (BitPointer>0)
             {
                 bytes = (BitPointer / 8);
                 j = (BitPointer % 8) ;
                 if ((j % 8) != 0 )
                 {
                     Data[bytes] = Data[bytes]<<(8-j);
                     bytes++;
                  }
                  fwrite(&Data,1,bytes,fp);
                  FileSize+=bytes;
              }
              fclose(fp);
              return (FileSize);
         }

};
// end WBuffer class

#endif // WBUFFER_HPP_