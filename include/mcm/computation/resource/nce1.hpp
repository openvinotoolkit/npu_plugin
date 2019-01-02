#ifndef NCE1_HPP
#define NCE1_HPP

#include <map>
#include <array>
#include "mcm/algorithms/dijkstra.hpp"

namespace mv
{

    enum NCE1Modes
    {
        Mode0 = 0,
        Mode1 = 1,
        Mode2 = 2,
        Mode3 = 3,
        Mode4 = 4
    };

    enum NCE1Splits
    {
        InputChannel,
        Width,
        NoSplit
    };

    enum NCE1HWOps
    {
        FullyConnected,
        Convolution,
        MaxPooling,
        AveragePooling
    };

    struct SplitOverHSolution
    {
        int input_lines_processed;
        int output_lines_processed;
        int junk_output_before;
        int junk_output_after;
        int start_input_line;
        int end_input_line;
        int start_output_line;
        int end_output_line;

        friend std::ostream& operator<<(std::ostream& os, const SplitOverHSolution sol)
        {
            os << "(" << sol.input_lines_processed << ", "
                         << sol.output_lines_processed << ", "
                         << sol.junk_output_before << ", "
                         << sol.junk_output_after << ", "
                         << sol.start_input_line << ", "
                         << sol.end_input_line << ", "
                         << sol.start_output_line << ", "
                         << sol.end_output_line << ")";
            return os;
        }
    };

    struct InputLinesPerOutputLinesSolution
    {
        int input_start_index;
        int input_end_index;
        int input_lines_before;
        int input_lines_after;
        int junk_output_lines_before;
        int junk_output_lines_after;

        friend std::ostream& operator<<(std::ostream& os, const InputLinesPerOutputLinesSolution sol)
        {
            os << "Input start index " << sol.input_start_index << std::endl;
            os << "Input end index " << sol.input_end_index << std::endl;
            os << "Input lines before " << sol.input_lines_before << std::endl;
            os << "Input lines after " << sol.input_lines_after << std::endl;
            os << "Junk output lines before " << sol.junk_output_lines_before << std::endl;
            os << "Junk output lines after " << sol.junk_output_lines_after << std::endl;
            return os;
        }
    };

    struct ConvolutionParameters
    {
        unsigned kernel_width;
        unsigned kernel_height;
        unsigned stride_vertical;
        unsigned stride_horizontal;
        unsigned input_channels;
        unsigned output_channels;
        unsigned input_width;
        unsigned input_height;
        unsigned output_width;
        unsigned output_height;
        unsigned pad_x_up;
        unsigned pad_x_down;
        unsigned pad_y_left;
        unsigned pad_y_right;

        ConvolutionParameters(
        unsigned kernel_width_param,
        unsigned kernel_height_param,
        unsigned stride_vertical_param,
        unsigned stride_horizontal_param,
        unsigned input_channels_param,
        unsigned output_channels_param,
        unsigned input_width_param,
        unsigned input_height_param,
        unsigned output_width_param,
        unsigned output_height_param,
        unsigned pad_x_up_param,
        unsigned pad_x_down_param,
        unsigned pad_y_left_param,
        unsigned pad_y_right_param)
            :kernel_width(kernel_width_param),
             kernel_height(kernel_height_param),
             stride_vertical(stride_vertical_param),
             stride_horizontal(stride_horizontal_param),
             input_channels(input_channels_param),
             output_channels(output_channels_param),
             input_width(input_width_param),
             input_height(input_height_param),
             output_width(output_width_param),
             output_height(output_height_param),
             pad_x_up(pad_x_up_param),
             pad_x_down(pad_x_down_param),
             pad_y_left(pad_y_left_param),
             pad_y_right(pad_y_right_param)
        {

        }

        ConvolutionParameters()
            :kernel_width(0),
             kernel_height(0),
             stride_vertical(0),
             stride_horizontal(0),
             input_channels(0),
             output_channels(0),
             input_width(0),
             input_height(0),
             output_width(0),
             output_height(0),
             pad_x_up(0),
             pad_x_down(0),
             pad_y_left(0),
             pad_y_right(0)
        {

        }

        ConvolutionParameters(const ConvolutionParameters& other)
            :kernel_width(other.kernel_width),
             kernel_height(other.kernel_height),
             stride_vertical(other.stride_vertical),
             stride_horizontal(other.stride_horizontal),
             input_channels(other.input_channels),
             output_channels(other.output_channels),
             input_width(other.input_width),
             input_height(other.input_height),
             output_width(other.output_width),
             output_height(other.output_height),
             pad_x_up(other.pad_x_up),
             pad_x_down(other.pad_x_down),
             pad_y_left(other.pad_y_left),
             pad_y_right(other.pad_y_right)
        {

        }
    };

    struct ModeSelectionNode
    {
        int remaining_output_channels;
        ConvolutionParameters parameters;

        ModeSelectionNode()
            :remaining_output_channels(0),
             parameters()
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
        NCE1Splits split_performed;

        ModeSelectionDistance()
        :
            cost(-1),
            mode(-1),
            performed_output_channels(-1),
            num_splits(-1),
            split_performed(NCE1Splits::NoSplit)
        {

        }

        ModeSelectionDistance(int n)
        :
            cost(n),
            mode(-1),
            performed_output_channels(-1),
            num_splits(-1),
            split_performed(NCE1Splits::NoSplit)
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
            unsigned max_size_true_adaptation_B0_pad_bottom;
            unsigned max_size_true_adaptation_B0_pad_left_right;

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
            unsigned computeMinLinesForConvolution(ConvolutionParameters param);

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
            ModeSelectionResult optimize_convolution(ModeSelectionNode source);
            ModeSelectionResult optimize_pooling(ModeSelectionNode source);

            //Constraint check functions
            //IMPORTANT: All the check functions must be invoked with param values already rounded up to the needed values.
            bool check_min_lines_constraint(unsigned kernel_height, unsigned stride_vertical, unsigned input_width, unsigned input_channels);
            bool check_coefficient_size_constraint(unsigned kernel_x, unsigned kernel_y, unsigned input_channels, unsigned output_channel_performed);
            bool check_coefficient_line_constraint(unsigned input_channels, unsigned kernel_x, unsigned kernel_height, int mode);
            bool check_channels_per_ram_block(unsigned input_channels, int mode);

            //Padding helper functions
            unsigned computeActualInputChannels(unsigned input_channels, unsigned mode);
            unsigned computeActualInputChannels(unsigned input_channels);
            unsigned computeActualInputWidth(unsigned input_width);
            unsigned computeActualInputHeight(unsigned input_height);
            unsigned computeActualInputChannelSplits(unsigned splits);
            unsigned computeActualOutputWidth(unsigned output_width);
            unsigned computerActualOutputHeight(unsigned output_height);
            unsigned computeActualOutputChannels(unsigned output_channels);

            //Splits over h
            std::vector<SplitOverHSolution> computeSplitsOverH(ConvolutionParameters param, unsigned max_output_lines);

            //Other helper functions
            unsigned computeLocalLineStride(unsigned input_width);
            unsigned computeDescriptorSplits(unsigned splits_over_height, unsigned splits_over_input_channels, unsigned num_modes);
            unsigned computeInputChannelsPerRamBlock(unsigned input_channels, unsigned mode);
            unsigned computeLinesPerChannel(unsigned input_channels, unsigned local_line_stride, unsigned mode);
            unsigned computeMaxOutputLinesConvolution(unsigned width, unsigned output_channel_performed);
            unsigned computeMaxOutputLinesPooling(unsigned width, unsigned output_channel_performed, std::array<unsigned short, 4> padding, std::array<unsigned short, 2> kernel);
            unsigned computeMinLinesForConvolution(unsigned kernel_height, unsigned stride_vertical);


            //Getter methods
            unsigned getBytesPerLine();
            unsigned getWordsPerLine();
            unsigned getMaxNumberOfLinesInDataStorage();

    };
}

#endif
