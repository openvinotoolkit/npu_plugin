#ifndef NCE1_HPP
#define NCE1_HPP

#include <map>
#include "mcm/algorithms/dijkstra.hpp"

namespace mv
{

    enum Modes
    {
        Mode0 = 0,
        Mode1 = 1,
        Mode2 = 2,
        Mode3 = 3,
        Mode4 = 4
    };

    enum Splits
    {
        InputChannel,
        Width,
        NoSplit
    };

    struct ConvolutionParameters
    {
        unsigned kernel_x;
        unsigned kernel_y;
        unsigned stride_x;
        unsigned stride_y;
        unsigned input_channels;
        unsigned output_channels;
        unsigned input_width;
        unsigned input_height;
        unsigned output_width;
        unsigned output_height;

        ConvolutionParameters(
        unsigned kernel_x_param,
        unsigned kernel_y_param,
        unsigned stride_x_param,
        unsigned stride_y_param,
        unsigned input_channels_param,
        unsigned output_channels_param,
        unsigned input_width_param,
        unsigned input_height_param,
        unsigned output_width_param,
        unsigned output_height_param)
            :kernel_x(kernel_x_param),
             kernel_y(kernel_y_param),
             stride_x(stride_x_param),
             stride_y(stride_y_param),
             input_channels(input_channels_param),
             output_channels(output_channels_param),
             input_width(input_width_param),
             input_height(input_height_param),
             output_width(output_width_param),
             output_height(output_height_param)
        {

        }

        ConvolutionParameters()
        {

        }

        ConvolutionParameters(const ConvolutionParameters& other)
            :kernel_x(other.kernel_x),
             kernel_y(other.kernel_y),
             stride_x(other.stride_x),
             stride_y(other.stride_y),
             input_channels(other.input_channels),
             output_channels(other.output_channels),
             input_width(other.input_width),
             input_height(other.input_height),
             output_width(other.output_width),
             output_height(other.output_height)
        {

        }
    };

    struct ModeSelectionNode
    {
        int remaining_output_channels;
        ConvolutionParameters parameters;

        ModeSelectionNode()
        {

        }

        ModeSelectionNode(int n)
            :remaining_output_channels(n)
        {

        }

        ModeSelectionNode(const ModeSelectionNode& other)
            :remaining_output_channels(other.remaining_output_channels),
             parameters(other.parameters)
        {

        }

        friend bool operator<(const ModeSelectionNode& a, const ModeSelectionNode& b)
        {
            return a.remaining_output_channels < b.remaining_output_channels;
        }

        friend bool operator==(const ModeSelectionNode& a, const ModeSelectionNode& b)
        {
            return a.remaining_output_channels == b.remaining_output_channels;
        }

        friend bool operator!=(const ModeSelectionNode& a, const ModeSelectionNode& b)
        {
            return !operator==(a, b);
        }
    };

    struct ModeSelectionDistance
    {
        int cost;
        int mode;
        int performed_output_channels;
        int num_splits;
        Splits split_performed;

        ModeSelectionDistance()
        :
            cost(-1),
            mode(-1),
            performed_output_channels(-1),
            num_splits(-1),
            split_performed(Splits::NoSplit)
        {

        }

        ModeSelectionDistance(int n)
        :
            cost(n),
            mode(-1),
            performed_output_channels(-1),
            num_splits(-1),
            split_performed(Splits::NoSplit)
        {

        }

        ModeSelectionDistance(const ModeSelectionDistance& other)
        :
            cost(other.cost),
            mode(other.mode),
            performed_output_channels(other.performed_output_channels),
            num_splits(other.num_splits),
            split_performed(other.split_performed)
        {

        }

        ModeSelectionDistance& operator=(const ModeSelectionDistance& other)
        {
            cost = other.cost;
            mode = other.mode;
            performed_output_channels = other.performed_output_channels;
            num_splits = other.num_splits;
            split_performed = other.split_performed;
            return *this;
        }

        friend bool operator<(const ModeSelectionDistance& a, const ModeSelectionDistance& b)
        {
            if(a.cost < 0) //Negative cost is actually infinite cost
                return false;
            if(b.cost < 0)
                return true;
            return a.cost < b.cost;
        }

        friend bool operator>(const ModeSelectionDistance& a, const ModeSelectionDistance& b)
        {
            if(a.cost < 0) //Negative cost is actually infinite cost
                return true;
            if(b.cost < 0)
                return false;
            return a.cost > b.cost;
        }

        friend ModeSelectionDistance operator+(const ModeSelectionDistance& a, const ModeSelectionDistance& b)
        {
            ModeSelectionDistance toReturn(b);
            toReturn.cost += a.cost;
            return toReturn;
        }
    };

    using ModeSelectionResult = DijkstraReturnValue<ModeSelectionNode, ModeSelectionDistance>;

    class Nce1
    {
        private:

            //Num of Data Path Elements
            unsigned nce1_dpe;

            //Input data size in bytes
            unsigned input_data_size;

            //Memory dimensions
            unsigned coefficients_storage_dimension;
            unsigned data_storage_dimension;
            unsigned cmx_stream_size;

            //Hw constraints
            unsigned max_coefficient_number_per_line;
            unsigned max_descriptors_x_hw_op;

            //Support data structures
            std::map<unsigned, unsigned> dpe_x_output_channel;
            std::map<unsigned, unsigned> output_channel_performed_one_shot;
            std::map<unsigned, unsigned> reverse_output_channel_performed_one_shot;

            //Split overheads
            unsigned split_by_input_channel_overhead;
            unsigned split_by_width_overhead;
            unsigned split_by_output_channel_overhead;

            //Ram features
            unsigned bits_per_line;

            //-----------PRIVATE FUNCTIONS-------------

            //Functions for Dijkstra
            std::vector<ModeSelectionNode> generateNeighboursComingFromValidModes(const ModeSelectionNode current_node);
            ModeSelectionDistance computeModeCost(const ModeSelectionNode a, const ModeSelectionNode b);

            //Split functions
            ModeSelectionDistance split_by_width(ConvolutionParameters param, int mode, bool support_split_over_w = true);
            ModeSelectionDistance split_by_input_channel(ConvolutionParameters param, unsigned actual_output_channels, int mode, bool support_split_over_c = true);

            //Utility functions
            unsigned get_max_mode(unsigned input_channels);
            std::vector<unsigned> get_valid_modes(ModeSelectionNode node);

            //Constraint check functions, private overload
            //IMPORTANT: All the check functions must be invoked with param values already rounded up to the needed values.
            bool check_min_lines_constraint_(ConvolutionParameters param);
            bool check_coefficient_size_constraint_(ConvolutionParameters param, unsigned output_channel_performed);
            bool check_coefficient_line_constraint_(ConvolutionParameters param, int mode);
            bool check_channels_per_ram_block_(ConvolutionParameters param, int mode);

        public:
            //Constructor
            Nce1();

            //Mode selection procedure
            ModeSelectionResult optimize_convolution(const ModeSelectionNode source);

            //Constraint check functions
            //IMPORTANT: All the check functions must be invoked with param values already rounded up to the needed values.
            bool check_min_lines_constraint(unsigned kernel_y, unsigned stride_y, unsigned input_width, unsigned input_channels);
            bool check_coefficient_size_constraint(unsigned kernel_x, unsigned kernel_y, unsigned input_channels, unsigned output_channel_performed);
            bool check_coefficient_line_constraint(unsigned input_channels, unsigned kernel_x, unsigned kernel_y, int mode);
            bool check_channels_per_ram_block(unsigned input_channels, int mode);
            unsigned getSplitsOverH(unsigned total_memory_occupied_by_tensors);

            //Padding helper functions
            unsigned getActualInputChannels(unsigned input_channels, unsigned mode);
            unsigned getActualInputChannels(unsigned input_channels);
            unsigned getActualInputWidth(unsigned input_width);
            unsigned getActualInputHeight(unsigned input_height);
            unsigned getActualInputChannelSplits(unsigned splits);

            unsigned getActualOutputWidth(unsigned output_width);
            unsigned getActualOutputHeight(unsigned output_height);
            unsigned getActualOutputChannels(unsigned output_channels);

            //Other helper functions
            unsigned computeLocalLineStride(unsigned input_width);
            unsigned computeDescriptorSplits(unsigned splits_over_height, unsigned splits_over_input_channels, float actual_output_channels, std::vector<unsigned>& modes);
            unsigned computeInputChannelsPerRamBlock(unsigned input_channels, unsigned mode);
            unsigned computeLinesPerChannel(unsigned channel_per_block, unsigned mode, unsigned local_line_stride);
            unsigned computeBytesPerLine();
            unsigned computePixelsPerLine();

    };
}

#endif
