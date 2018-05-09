/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle
* @date 4/27/2018
*/

#include <string>
#include "graph.hpp"

namespace mv
{
/// List of supported node serialization formats
enum serializer_mode
{
    mvblob_mode,
    json_mode,
    flatbuffers_mode
};

class Blob_stage
{
    public:
        uint32_t next ;
        uint32_t op_type ;
        uint32_t implementation  ;
        uint32_t preop_type  ;
        uint32_t postop_type ;
 
        Blob_stage()
        {
            next = 0x0000 ;
            op_type = 0xeeee;
            implementation = 0xffff ;
            preop_type = 0xaaaa ;
            postop_type = 0xbbbb ;
        }
};

class WBuffer 
{
    private:
        char Data[4096] ;
        int BitPointer ;

    public:
        unsigned long int FileSize ;
        WBuffer()
        {
            BitPointer  = 0 ;
            FileSize = 0 ;
        }

        uint32_t align(uint32_t number_2_round, uint32_t align_size)
        {
            return align_size+(number_2_round/align_size)*align_size ;
        }

        void write_elf_header()
        {
            int j;
            const int elfhdr_length = 34 ;

            for (j=0; j< elfhdr_length; j++)
                {
                //    Data[j] = 0x00 ;   // Zero all fields of 34 byte ELF header
                    Data[j] =j ;   // Zero all fields of 34 byte ELF header
                }

/* temporarily disable elf header
            // E_IDENT
            Data[0] = 0x7f ;   // EI_MAG = x7f.ELF
            Data[1] = 'E' ; 
            Data[2] = 'L' ;
            Data[3] = 'F' ;
            Data[4] = 0x01 ;    // EI_CLASS = 1
            Data[5] = 0x01 ;    // EI_DATA  = 1
            Data[6] = 0x01 ;    // EI_VERSION = 1
                                // EI_OSABI, EI_ABIVERSION, EI_PAD = 0


            Data[17] = 0x01 ;    // E_TYPE = 1
            Data[19] = 0x02 ;    // E_MACHINE = 2
            Data[21] = 0x01 ;    // E_VERSION = 1
                                 // E_ENTRY, E_PHOFF, E_SHOFF, E_FLAGS = 0
            Data[37] = 0x30 ;    // E_EHSIZE = 48 (0x30)
*/
            BitPointer = 8 * elfhdr_length ;

        }

       void write_mv_header(FILE *fp)
        {
            uint32_t elf_header_size = 34; 
            uint32_t mv_header_size = 40; 
            uint32_t total_header_size = elf_header_size+mv_header_size; 

            uint32_t mv_magic_number = 8708 ;
            uint32_t mv_filesize = 0xdead ;
            uint32_t mv_version_major = 2 ;
            uint32_t mv_version_minor = 4 ;
            uint32_t mv_num_shaves = 1 ;

            uint32_t mv_stage_section_offset = align(total_header_size,0x10) ;
            uint32_t mv_header_pad_size = mv_stage_section_offset - total_header_size ; 

            uint32_t mv_buffer_section_offset = 0xbeef ;
            uint32_t mv_relocation_offset = 0xa5a5 ;
            uint32_t mv_size_of_input = 0xc3c3 ;
            uint32_t mv_permutation_enabled = 0xface ;
            
            AddBytes(4, mv_magic_number, fp);

            AddBytes(4, mv_filesize, fp);
            AddBytes(4, mv_version_major, fp);
            AddBytes(4, mv_version_minor, fp);
            AddBytes(4, mv_num_shaves, fp);
            AddBytes(4, mv_stage_section_offset, fp);
            AddBytes(4, mv_buffer_section_offset, fp);
            AddBytes(4, mv_relocation_offset, fp);
            AddBytes(4, mv_size_of_input, fp);
            AddBytes(4, mv_permutation_enabled, fp);

            AddBytes(mv_header_pad_size, 0x1234, fp);

        }

       void write_stage_header(FILE *fp)
       {
            uint32_t stage_count = 1;
            uint32_t stage_section_size = 0xdead ;
            uint32_t output_size = 0xbeef ;

            AddBytes(4, stage_count, fp);
            AddBytes(4, stage_section_size, fp);
            AddBytes(4, output_size, fp);

       }

       void write_stage_section(FILE *fp)
       {

            Blob_stage test_1conv_stage ;

            // stage section header
            AddBytes(4, test_1conv_stage.next, fp);
            AddBytes(4, test_1conv_stage.op_type, fp);
            AddBytes(4, test_1conv_stage.implementation, fp);

            for (int i=0; i<48; i++)
            {
            AddBytes(4, 0xdddddddd, fp);
            }
/*
            // operator specific info
            AddBytes(4, test_1Conv.radixX, fp);
            AddBytes(4, test_1Conv.radixY, fp);
            AddBytes(4, test_1Conv.radixStrideX, fp);
            AddBytes(4, test_1Conv.radixStrideY, fp);
            AddBytes(4, test_1Conv.padX, fp);
            AddBytes(4, test_1Conv.padY, fp);
            AddBytes(4, test_1Conv.padStyle, fp);
            AddBytes(4, test_1Conv.dilation, fp);

            // python helper push
            AddBytes(4, test_1Conv.InputDimX, fp);
            AddBytes(4, test_1Conv.InputDimY, fp);
            AddBytes(4, test_1Conv.InputDimZ, fp);
            AddBytes(4, test_1Conv.InputStrideX, fp);
            AddBytes(4, test_1Conv.InputStrideY, fp);
            AddBytes(4, test_1Conv.InputStrideZ, fp);
            AddBytes(4, test_1Conv.InputOffset, fp);
            AddBytes(4, test_1Conv.InputLocation, fp);
            AddBytes(4, test_1Conv.InputDataType, fp);
            AddBytes(4, test_1Conv.InputOrder, fp);
            //
            AddBytes(4, test_1Conv.OutputDimX, fp);
            AddBytes(4, test_1Conv.OutputDimY, fp);
            AddBytes(4, test_1Conv.OutputDimZ, fp);
            AddBytes(4, test_1Conv.OutputStrideX, fp);
            AddBytes(4, test_1Conv.OutputStrideY, fp);
            AddBytes(4, test_1Conv.OutputStrideZ, fp);
            AddBytes(4, test_1Conv.OutputOffset, fp);
            AddBytes(4, test_1Conv.OutputLocation, fp);
            AddBytes(4, test_1Conv.OutputDataType, fp);
            AddBytes(4, test_1Conv.OutputOrder, fp);
            //
            AddBytes(4, test_1Conv.TapsDimX, fp);
            AddBytes(4, test_1Conv.TapsDimY, fp);
            AddBytes(4, test_1Conv.TapsDimZ, fp);
            AddBytes(4, test_1Conv.TapsStrideX, fp);
            AddBytes(4, test_1Conv.TapsStrideY, fp);
            AddBytes(4, test_1Conv.TapsStrideZ, fp);
            AddBytes(4, test_1Conv.TapsOffset, fp);
            AddBytes(4, test_1Conv.TapsLocation, fp);
            AddBytes(4, test_1Conv.TapsDataType, fp);
            AddBytes(4, test_1Conv.TapsOrder, fp);
            //
            AddBytes(4, test_1Conv.BiasDimX, fp);
            AddBytes(4, test_1Conv.BiasDimY, fp);
            AddBytes(4, test_1Conv.BiasDimZ, fp);
            AddBytes(4, test_1Conv.BiasStrideX, fp);
            AddBytes(4, test_1Conv.BiasStrideY, fp);
            AddBytes(4, test_1Conv.BiasStrideZ, fp);
            AddBytes(4, test_1Conv.BiasOffset, fp);
            AddBytes(4, test_1Conv.BiasLocation, fp);
            AddBytes(4, test_1Conv.BiasDataType, fp);
            AddBytes(4, test_1Conv.BiasOrder, fp);
*/
            AddBytes(4, test_1conv_stage.preop_type, fp);
            AddBytes(4, test_1conv_stage.postop_type, fp);


            uint32_t total_stage_size = (0x50+(8*4+48*4)) ;
            uint32_t buffer_section_offset = align(total_stage_size,0x10) ;
            uint32_t stage_pad_size = buffer_section_offset - total_stage_size ;

            std::cout << "total_size before align = "<< total_stage_size << std::endl;
            std::cout << "buffer section offset = "<< buffer_section_offset << std::endl;

            AddBytes(stage_pad_size, 0x87654321, fp);
        }

       void write_buffer_section(FILE *fp)
       {
            uint32_t buffer_size = 0x200;
            uint32_t buffer_pad_val = 0x002a ;
            uint8_t buffer_data_val = 0xFF ;

            AddBytes(4, buffer_size, fp);

            for (int i=0; i<3; i++)
            {   
                AddBytes(4, buffer_pad_val, fp);
            }

            // data buffer
            for (int i=0; i<buffer_size; i++) 
            {
                AddBytes(1, buffer_data_val, fp);
            }

       }

       void write_relocation_section(FILE *fp)
       {

            uint32_t relocation_section_size = 0x1c;
            uint32_t blob_buffer_reloc_offset = 0x364;
            uint32_t blob_buffer_reloc_size = 0x8;
            uint32_t work_buffer_reloc_offset= 0x36c;
            uint32_t work_buffer_reloc_size = 0x00;
            uint32_t work_link_off = 0x00;
            uint32_t work_link_size = 0x03;

            AddBytes(4,relocation_section_size , fp);
            AddBytes(4, blob_buffer_reloc_offset, fp);
            AddBytes(4, blob_buffer_reloc_size, fp);
            AddBytes(4, work_buffer_reloc_offset, fp);
            AddBytes(4, work_buffer_reloc_size, fp);
            AddBytes(4, work_link_off, fp);
            AddBytes(4, work_link_size, fp);

        }

        void AddBytes(int numbytes, uint32_t field, FILE *fp)
        {
            int byte_pointer ;
            int j;

            byte_pointer = (BitPointer / 8);

            // dump buffer if full 
            if ((numbytes*8+BitPointer) > 4096-300)
            {
                fwrite(&Data,1,byte_pointer,fp);
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
 
        void AddBits(int numbits, int index, FILE *fp)
        {
            int bytes ;
            int j;
            char thisbit ;

            bytes = (BitPointer / 8);

            // dump buffer if full 
            if ((numbits+BitPointer) > 4096-300)
            {
                fwrite(&Data,1,bytes,fp);
                BitPointer = BitPointer - (8*bytes);
                Data[0]=Data[bytes] ;
                Data[1]=Data[bytes+1] ;
                bytes = BitPointer / 8;
            }

            // write IndexLen bits to output buffer 
            for (j=(numbits-1); j>=0; j--)
            {
                thisbit = ((index>>j) & (0x01)) ;
                Data[bytes]=Data[bytes]<<1;
                Data[bytes]=Data[bytes] | thisbit ;
                BitPointer++;
                if ((BitPointer % 8) == 0)
                {
                    bytes++;
                    FileSize++;
                }
             }

         }

         void End(FILE *fp)
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
              }
              fclose(fp) ;
         }
     };

/**
* @brief Serializer outputs verious representations of the compute graph. Initially moviduius binary blob format is supported.
*
* @param set_serialization_mode defines the output format of the graph
*/
    class Serializer 
    {

    private:
        serializer_mode output_format;
        FILE *fp ;
        char const *out_file_name;
        WBuffer odata;

    public:
  
        Serializer(serializer_mode set_output_format) 
        {
            output_format = set_output_format;
            out_file_name = "test_output.txt";
        }

/**
* @brief write_blob writes the blob format output file desecribing the computre model.
*
* @param graph_2_show (by reference) points to the graph you want to visualize
*/
        template <class T_node, class T_edge>
        void write_blob(mv::graph<T_node, T_edge, mv::stl_allocator >& graph_2_show )
        {
            // open output file
            if ((fp = fopen(out_file_name, "w")) == NULL)
            {   
                std::cout << "ERROR: Could not open output filei" << std::endl;
            }
            else std::cout <<"Opened output file" << std::endl;
 
            odata.write_elf_header();
            odata.write_mv_header(fp);
            odata.write_stage_header(fp);
            odata.write_stage_section(fp);
            odata.write_buffer_section(fp);
            odata.write_relocation_section(fp);

            // close output file
            odata.End(fp) ;
        }

        void print_mode()
        {
            std::cout << "serializer output mode= " << output_format << std::endl;
        }

    };

}
