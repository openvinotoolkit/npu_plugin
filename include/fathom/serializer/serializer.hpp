/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle
* @date 4/27/2018
*/

#include "graph.hpp"

namespace mv
{
/// List of supported node serialization formats
enum serializer_mode
{
    mvblob_mode,
    json_mode,
    flatbuffers_mode,
    dot_mode
};

// Generic 4KB output buffer supporting bit-level output to file.
// Buffer empties at 3800 level. Assumes add size < 296 to prevent
// excessive checking during adds.

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


class Blob_stage
{
    public:
        uint32_t next ;
        uint32_t op_type ;
        uint32_t implementation  ;
        uint32_t preop_type  ;
        uint32_t postop_type ;

        uint32_t radixX;
        uint32_t radixY;
        uint32_t radixStrideX;
        uint32_t radixStrideY;
        uint32_t padX;
        uint32_t padY;
        uint32_t padStyle;
        uint32_t dilation;

        uint32_t InputDimX;
        uint32_t InputDimY;
        uint32_t InputDimZ;
        uint32_t InputStrideX;
        uint32_t InputStrideY;
        uint32_t InputStrideZ;
        uint32_t InputOffset;
        uint32_t InputLocation;
        uint32_t InputDataType;
        uint32_t InputOrder;

        uint32_t OutputDimX;
        uint32_t OutputDimY;
        uint32_t OutputDimZ;
        uint32_t OutputStrideX;
        uint32_t OutputStrideY;
        uint32_t OutputStrideZ;
        uint32_t OutputOffset;
        uint32_t OutputLocation;
        uint32_t OutputDataType;
        uint32_t OutputOrder;

        uint32_t TapsDimX;
        uint32_t TapsDimY;
        uint32_t TapsDimZ;
        uint32_t TapsStrideX;
        uint32_t TapsStrideY;
        uint32_t TapsStrideZ;
        uint32_t TapsOffset;
        uint32_t TapsLocation;
        uint32_t TapsDataType;
        uint32_t TapsOrder;

        uint32_t BiasDimX;
        uint32_t BiasDimY;
        uint32_t BiasDimZ;
        uint32_t BiasStrideX;
        uint32_t BiasStrideY;
        uint32_t BiasStrideZ;
        uint32_t BiasOffset;
        uint32_t BiasLocation;
        uint32_t BiasDataType;
        uint32_t BiasOrder;

        Blob_stage()
        {
            next = 0x0000 ;
            op_type = 0x0000;
            implementation = 0x80000000 ;

            radixX = 3 ;
            radixY = 3 ;
            radixStrideX = 2 ;
            radixStrideY = 2 ;
            padX = 0 ;
            padY = 0 ;
            padStyle = 3 ;
            dilation = 1 ;

            InputDimX = 32 ;
            InputDimY = 32 ;
            InputDimZ = 3 ;
            InputStrideX = 6 ;
            InputStrideY = 0xc0 ;
            InputStrideZ = 2 ;
            InputOffset = 0 ;
            InputLocation = 1 ;
            InputDataType = 0 ;
            InputOrder = 0 ;

            OutputDimX = 16 ;
            OutputDimY = 16 ;
            OutputDimZ = 8 ;
            OutputStrideX = 16 ;
            OutputStrideY = 0x100 ;
            OutputStrideZ = 2 ;
            OutputOffset = 0 ;
            OutputLocation = 2 ;
            OutputDataType = 0 ;
            OutputOrder = 0 ;

            TapsDimX = 9 ;
            TapsDimY = 3 ;
            TapsDimZ = 8 ;
            TapsStrideX = 48 ;
            TapsStrideY = 16 ;
            TapsStrideZ = 2 ;
            TapsOffset = 0 ;
            TapsLocation = 3 ;
            TapsDataType = 0 ;
            TapsOrder = 3 ;

            BiasDimX = 0 ;
            BiasDimY = 0 ;
            BiasDimZ = 0 ;
            BiasStrideX = 0 ;
            BiasStrideY = 0 ;
            BiasStrideZ = 0 ;
            BiasOffset = 0 ;
            BiasLocation = 0 ;
            BiasDataType = 0 ;
            BiasOrder = 1 ;

            preop_type = 5 ;
            postop_type = 5 ;
        }
};

struct blob_summary {
    uint32_t elf_header_size;
    uint32_t mv_header_size; 
    uint32_t header_pad_size; 
    uint32_t stage_section_size; 
    uint32_t buffer_header_size; 
    uint32_t buffer_data_size; 
    uint32_t relocation_section_size; 
    uint32_t stage_count;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t blob_file_size; 
};


class Blob_buffer : public WBuffer
{
    private:
        blob_summary blob_stats;

    public:
        template <class T_node, class T_edge>
        void calc(mv::graph<T_node, T_edge, mv::stl_allocator >& graph_2_show)
        {
            blob_stats.elf_header_size = 34 ;
            blob_stats.mv_header_size = 40 ;
            uint32_t headers_data_size = blob_stats.elf_header_size+blob_stats.mv_header_size ;
            blob_stats.header_pad_size = align(headers_data_size,0x10)-headers_data_size;
            blob_stats.stage_section_size = 0xf0 ;
            blob_stats.buffer_header_size = 0x10 ;
            blob_stats.buffer_data_size = 0x200 ;
            blob_stats.relocation_section_size = 0x1c ;
            blob_stats.stage_count = 1 ;
            blob_stats.input_size = 0x0c00 ;
            blob_stats.output_size = 0x800 ;
            blob_stats.blob_file_size = headers_data_size+blob_stats.header_pad_size+blob_stats.stage_section_size+blob_stats.buffer_header_size+blob_stats.buffer_data_size+blob_stats.relocation_section_size ;
        }

        void write_elf_header()
        {
            int j;
            const int elfhdr_length = 34 ;

            for (j=0; j< elfhdr_length; j++)
                {
                   AddBytes(1, 0x00);
                }

/* temporarily disable elf header
            // E_IDENT
            AddBytes(4, 0x464c457f);// EI_MAG = x7f.ELF
            AddBytes(1, 0x01) ;     // EI_CLASS = 1
            AddBytes(1, 0x01) ;     // EI_DATA  = 1
            AddBytes(1, 0x01) ;     // EI_VERSION = 1
            AddBytes(10, 0x00) ;    // EI_OSABI, EI_ABIVERSION, EI_PAD = 0
            AddBytes(2, 0x0001) ;   // E_TYPE = 1
            AddBytes(2, 0x0002) ;   // E_MACHINE = 2
            AddBytes(1, 0x01);      // E_VERSION = 1
            AddBytes(15, 0x00) ;    // E_ENTRY, E_PHOFF, E_SHOFF, E_FLAGS = 0
            AddBytes(1, 0x30 ;      // E_EHSIZE = 48 (0x30)
            AddBytes(10, 0x00) ;    // pad
*/

        }

       void write_mv_header()
        {
            uint32_t total_header_size = blob_stats.elf_header_size+blob_stats.mv_header_size; 

            uint32_t mv_magic_number = 8708 ;
            uint32_t mv_version_major = 2 ;
            uint32_t mv_version_minor = 4 ;
            uint32_t mv_num_shaves = 1 ;

            uint32_t mv_stage_section_offset = blob_stats.elf_header_size+blob_stats.mv_header_size+blob_stats.header_pad_size ;

            uint32_t mv_buffer_section_offset = mv_stage_section_offset + blob_stats.stage_section_size ;
            uint32_t mv_relocation_offset = mv_buffer_section_offset + blob_stats.buffer_header_size + blob_stats.buffer_data_size ;
            uint32_t mv_permutation_enabled = 0x0000 ;
            
            AddBytes(4, mv_magic_number);

            AddBytes(4, blob_stats.blob_file_size);
            AddBytes(4, mv_version_major);
            AddBytes(4, mv_version_minor);
            AddBytes(4, mv_num_shaves);
            AddBytes(4, mv_stage_section_offset);
            AddBytes(4, mv_buffer_section_offset);
            AddBytes(4, mv_relocation_offset);
            AddBytes(4, blob_stats.input_size);
            AddBytes(4, mv_permutation_enabled);

            AddBytes(blob_stats.header_pad_size, 0x00);

        }

       void write_stage_section_header()
       {
            AddBytes(4, blob_stats.stage_count);
            AddBytes(4, blob_stats.stage_section_size);
            AddBytes(4, blob_stats.output_size);
       }

       void write_stage()
       {

            Blob_stage test_1conv_stage ;

            // this stage header
            AddBytes(4, test_1conv_stage.next);
            AddBytes(4, test_1conv_stage.op_type);
            AddBytes(4, test_1conv_stage.implementation);

            // operator specific info
            AddBytes(4, test_1conv_stage.radixX);
            AddBytes(4, test_1conv_stage.radixY);
            AddBytes(4, test_1conv_stage.radixStrideX);
            AddBytes(4, test_1conv_stage.radixStrideY);
            AddBytes(4, test_1conv_stage.padX);
            AddBytes(4, test_1conv_stage.padY);
            AddBytes(4, test_1conv_stage.padStyle);
            AddBytes(4, test_1conv_stage.dilation);

            // python helper push
            AddBytes(4, test_1conv_stage.InputDimX);
            AddBytes(4, test_1conv_stage.InputDimY);
            AddBytes(4, test_1conv_stage.InputDimZ);
            AddBytes(4, test_1conv_stage.InputStrideX);
            AddBytes(4, test_1conv_stage.InputStrideY);
            AddBytes(4, test_1conv_stage.InputStrideZ);
            AddBytes(4, test_1conv_stage.InputOffset);
            AddBytes(4, test_1conv_stage.InputLocation);
            AddBytes(4, test_1conv_stage.InputDataType);
            AddBytes(4, test_1conv_stage.InputOrder);

            AddBytes(4, test_1conv_stage.OutputDimX);
            AddBytes(4, test_1conv_stage.OutputDimY);
            AddBytes(4, test_1conv_stage.OutputDimZ);
            AddBytes(4, test_1conv_stage.OutputStrideX);
            AddBytes(4, test_1conv_stage.OutputStrideY);
            AddBytes(4, test_1conv_stage.OutputStrideZ);
            AddBytes(4, test_1conv_stage.OutputOffset);
            AddBytes(4, test_1conv_stage.OutputLocation);
            AddBytes(4, test_1conv_stage.OutputDataType);
            AddBytes(4, test_1conv_stage.OutputOrder);

            AddBytes(4, test_1conv_stage.TapsDimX);
            AddBytes(4, test_1conv_stage.TapsDimY);
            AddBytes(4, test_1conv_stage.TapsDimZ);
            AddBytes(4, test_1conv_stage.TapsStrideX);
            AddBytes(4, test_1conv_stage.TapsStrideY);
            AddBytes(4, test_1conv_stage.TapsStrideZ);
            AddBytes(4, test_1conv_stage.TapsOffset);
            AddBytes(4, test_1conv_stage.TapsLocation);
            AddBytes(4, test_1conv_stage.TapsDataType);
            AddBytes(4, test_1conv_stage.TapsOrder);

            AddBytes(4, test_1conv_stage.BiasDimX);
            AddBytes(4, test_1conv_stage.BiasDimY);
            AddBytes(4, test_1conv_stage.BiasDimZ);
            AddBytes(4, test_1conv_stage.BiasStrideX);
            AddBytes(4, test_1conv_stage.BiasStrideY);
            AddBytes(4, test_1conv_stage.BiasStrideZ);
            AddBytes(4, test_1conv_stage.BiasOffset);
            AddBytes(4, test_1conv_stage.BiasLocation);
            AddBytes(4, test_1conv_stage.BiasDataType);
            AddBytes(4, test_1conv_stage.BiasOrder);

            AddBytes(4, test_1conv_stage.preop_type);
            AddBytes(4, test_1conv_stage.postop_type);

            uint32_t total_stage_size = (0x50+(8*4+48*4)) ;
            uint32_t buffer_section_offset = align(total_stage_size,0x10) ;
            uint32_t stage_pad_size = buffer_section_offset - total_stage_size ;

            AddBytes(stage_pad_size, 0x00);
        }

       void write_buffer_section()
       {
            uint32_t buffer_header_pad_size = 3 ;
            uint32_t buffer_header_pad_val = 0x002a ;
            uint8_t buffer_data_val = 0xFF ;

            AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));

            for (int i=0; i<buffer_header_pad_size; i++)
            {   
                AddBytes(4, buffer_header_pad_val);
            }

            // data buffer
            for (int i=0; i< blob_stats.buffer_data_size; i++) 
            {
                AddBytes(1, buffer_data_val);
            }

       }

       void write_relocation_section()
       {

            uint32_t relocation_section_size = 0x1c;
            uint32_t blob_buffer_reloc_offset = 0x364;
            uint32_t blob_buffer_reloc_size = 0x8;
            uint32_t work_buffer_reloc_offset= 0x36c;
            uint32_t work_buffer_reloc_size = 0x00;
            uint32_t work_link_off = 0x00;
            uint32_t work_link_size = 0x03;

            AddBytes(4,relocation_section_size );
            AddBytes(4, blob_buffer_reloc_offset);
            AddBytes(4, blob_buffer_reloc_size);
            AddBytes(4, work_buffer_reloc_offset);
            AddBytes(4, work_buffer_reloc_size);
            AddBytes(4, work_link_off);
            AddBytes(4, work_link_size);

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
        Blob_buffer odata;

    public:
  
        Serializer(serializer_mode set_output_format) 
        {
            output_format = set_output_format;
        }

/**
* @brief serialize writes the blob format output file desecribing the computre model.
*
* @param graph_2_show (by reference) points to the graph you want to visualize
*/
        template <class T_node, class T_edge>
        uint64_t serialize(mv::graph<T_node, T_edge, mv::stl_allocator >& graph_2_show, const char* ofilename )
        {
            uint64_t fsize = 0 ;
            switch( output_format )
            {
                case mvblob_mode:
                    // calculate sizes and offsets for headers
                    odata.calc(graph_2_show);
                    // write to file
                    odata.open(ofilename);
                    odata.write_elf_header();
                    odata.write_mv_header();
                    odata.write_stage_section_header();
                    odata.write_stage();  //TODO temp, only one stage
                    odata.write_buffer_section();
                    odata.write_relocation_section();
                    fsize = odata.End() ;
                break;
                default:
                    std::cout << "ERROR: unsupported output format " << output_format << std::endl; 
                break;
            }
            return (fsize);
        }

        void print_mode()
        {
            std::cout << "serializer output mode= " << output_format << std::endl;
        }

};

}
