#ifndef MV_SERIALIZER_HPP_
#define MV_SERIALIZER_HPP_

/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle, Ian Hunter
* @date 4/27/2018
*/
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/deployer/Fp16Convert.h"
#include "include/mcm/deployer/file_buffer.h"
#include "include/mcm/deployer/blob_serializer.hpp"

namespace mv
{

    /// List of supported graph serialization formats
    enum serializer_mode
    {
        mvblob_mode,
        json_mode,
        flatbuffers_mode,
        dot_mode
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

            Serializer(serializer_mode set_output_format);

        /**
        * @brief serialize writes the specified format output file desecribing the compute model.
        *
        * @param graph_2_deploy (by reference) points to the graph you want to deploy
        */
        uint64_t serialize(mv::ControlModel& graph_2_deploy, const char* ofilename );

        void print_mode();
    };

}

#endif // MV_SERIALIZER_HPP_