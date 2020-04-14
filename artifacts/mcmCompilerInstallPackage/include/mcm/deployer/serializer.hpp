#ifndef MV_SERIALIZER_HPP_
#define MV_SERIALIZER_HPP_
/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle, Ian Hunter
* @date 4/27/2018
*/
#include <memory>
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/op_model.hpp"

#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv
{

    /**
    * @brief Serializer outputs verious representations of the compute graph. Initially moviduius binary blob format is supported.
    *
    * @param set_serialization_mode defines the output format of the graph
    */
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
        unsigned long long serialize(mv::ComputationModel& model, mv::TargetDescriptor& td );

        void print_mode();
    };

}

#endif // MV_SERIALIZER_HPP_
