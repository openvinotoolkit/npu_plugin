#include "mcm/graph/dijkstra.hpp"
#include <cmath>

const int nce1_dpe = 256;
const std::map<int, int> dpe_x_output_channel =
{
    {0, 1},
    {1, 2},
    {2, 4},
    {3, 8},
    {4, 16}
};

const std::map<int,int>& ram_blocks_x_mode = dpe_x_output_channel;
const std::map<int,int>& input_streams_x_mode = dpe_x_output_channel;

const std::map<int, int> output_channel_performed_one_shot =
{
    {0, 256},
    {1, 128},
    {2, 64},
    {3, 32},
    {4, 16}
};

const std::map<int, int> reverse_output_channel_performed_one_shot =
{
    {256, 0},
    {128, 1},
    {64, 2},
    {32, 3},
    {16, 4}
};

const int max_descriptors_x_hw_op = 255;

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
    OutputChannel,
    InputChannel,
    Width,
    NoSplit
};

struct ConvolutionParameters
{
    int kernel_x;
    int kernel_y;
    int stride_x;
    int stride_y;
    int input_channels;
    int output_channels;
    int width;
    int height;

    ConvolutionParameters()
    {

    }
};

struct ModeSelectionNode
{
    int remaining_output_channels;
    int mode_that_brings_here; //useful when remaining_output_channels == 0
    ConvolutionParameters parameters;

    ModeSelectionNode()
    {

    }

    ModeSelectionNode(int n)
        :remaining_output_channels(n),
         mode_that_brings_here(-1)
    {

    }

    ModeSelectionNode(const ModeSelectionNode& other)
        :remaining_output_channels(other.remaining_output_channels),
         mode_that_brings_here(other.mode_that_brings_here),
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

unsigned round_up(unsigned x, unsigned mult)
{
    return ((x + mult - 1) / mult) * mult;
}

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
        return a.cost < b.cost;
    }

    friend bool operator>(const ModeSelectionDistance& a, const ModeSelectionDistance& b)
    {
        return a.cost > b.cost;
    }

    friend ModeSelectionDistance operator+(const ModeSelectionDistance& a, const ModeSelectionDistance& b)
    {
        ModeSelectionDistance toReturn(b);
        toReturn.cost += a.cost;
        return toReturn;
    }
};

std::vector<ModeSelectionNode> generateNeighboursComingFromValidModes(ModeSelectionNode current_node)
{
    std::set<ModeSelectionNode> valid_neighbours_set;
    int max_mode = ceil(log2(current_node.parameters.input_channels)); //At least one channel per ramblock
    for(int mode = Mode0; mode <= Mode4; ++mode)
    {
        if(mode > max_mode)
            continue;
        int number_of_descriptors_needed = ceil((double)(current_node.remaining_output_channels) / output_channel_performed_one_shot.at(mode));
        if(number_of_descriptors_needed >= max_descriptors_x_hw_op) //Useless check with the current numbers, but you never know
            continue;
        ModeSelectionNode neighbour(current_node);
        neighbour.remaining_output_channels -= output_channel_performed_one_shot.at(mode);
        if(neighbour.remaining_output_channels < 0)
        {
            neighbour.remaining_output_channels = 0;
            neighbour.mode_that_brings_here = mode;
        }
        auto findIterator = valid_neighbours_set.find(neighbour);
        if(findIterator != valid_neighbours_set.end())
            valid_neighbours_set.erase(findIterator);
        valid_neighbours_set.insert(neighbour);
    }
    return std::vector<ModeSelectionNode>(valid_neighbours_set.begin(), valid_neighbours_set.end());
}

bool check_min_lines_constraint(ConvolutionParameters param)
{
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    //Space required by min lines = min_lines * input_width (rounded up to 16) * input data type size * num input channels
    unsigned space_required = min_lines * round_up(param.width, 16) * 2 * param.input_channels;
    return (space_required > pow(2, 17));
}

bool check_coefficient_size_constraint(ConvolutionParameters param, unsigned output_channel_performed)
{
    unsigned coeff_size = param.kernel_x * param.kernel_y*param.input_channels*round_up(output_channel_performed,8)*2;
    return (coeff_size > pow(2, 17));
}

bool check_coefficient_line_constraint(ConvolutionParameters param, int mode)
{
   // Calculate ram blocks
   unsigned channel_per_ramblock = param.input_channels / ram_blocks_x_mode.at(mode);
   //coefficient number per line must be lower than 256
   unsigned check_coeff_line_per_block = param.kernel_x*param.kernel_y*channel_per_ramblock;
   return (check_coeff_line_per_block > 256);
}

bool check_channels_per_ram_block(ConvolutionParameters param, int mode)
{
    return dpe_x_output_channel.at(mode) > param.input_channels;
}

ModeSelectionDistance split_by_input_channel(ConvolutionParameters param, int mode)
{

}

ModeSelectionDistance split_by_width(ConvolutionParameters param, int mode)
{

}

ModeSelectionDistance computeModeCost(const ModeSelectionNode a, const ModeSelectionNode b)
{
    ModeSelectionDistance to_return;

    int overhead = 0;
    int mode = 0;
    unsigned output_channel_performed = 0;

    bool need_split_by_k = false;
    if(b.remaining_output_channels == 0)
    {
        mode = b.mode_that_brings_here;
        need_split_by_k = false;
        output_channel_performed = a.remaining_output_channels;
    }
    else
    {
        mode = reverse_output_channel_performed_one_shot.at(a.remaining_output_channels - b.remaining_output_channels);
        need_split_by_k = true;
        output_channel_performed = output_channel_performed_one_shot.at(mode);
    }

    ConvolutionParameters parameters = a.parameters;
    int ram_blocks = ram_blocks_x_mode.at(mode);
    parameters.input_channels = round_up(parameters.input_channels, ram_blocks);

    bool need_split_by_width_or_input_channel = check_min_lines_constraint(parameters);
    bool need_split_by_input_channel = check_coefficient_size_constraint(parameters, output_channel_performed) | check_coefficient_line_constraint(parameters, mode);

    if(need_split_by_width_or_input_channel)
    {
        ModeSelectionDistance splitted_by_input_channel = split_by_input_channel(parameters, mode);
        ModeSelectionDistance splitted_by_width = split_by_width(parameters, mode);
        if(splitted_by_input_channel < splitted_by_width)
            to_return = splitted_by_input_channel;
        else
            to_return = splitted_by_width;
    }
    else if(need_split_by_input_channel)
        to_return = split_by_input_channel(parameters, mode);
    else
    {
        to_return.cost = parameters.kernel_x * parameters.kernel_y * parameters.width * parameters.height * parameters.input_channels / dpe_x_output_channel.at(mode);
        to_return.split_performed = Splits::NoSplit;
        to_return.num_splits = 1;
        if(check_channels_per_ram_block(parameters, mode))
            to_return.cost = -1; //infinite cost
    }

    if(need_split_by_k)
        to_return.cost += overhead;

    to_return.mode = mode;
    return to_return;
}
