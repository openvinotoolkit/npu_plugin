#include "mcm/graph/dijkstra.hpp"
#include <cmath>

const unsigned nce1_dpe = 256;
const std::map<unsigned, unsigned> dpe_x_output_channel =
{
    {0, 1},
    {1, 2},
    {2, 4},
    {3, 8},
    {4, 16}
};

const std::map<unsigned,unsigned>& ram_blocks_x_mode = dpe_x_output_channel;
const std::map<unsigned,unsigned>& input_streams_x_mode = dpe_x_output_channel;

const std::map<unsigned, unsigned> output_channel_performed_one_shot =
{
    {0, 256},
    {1, 128},
    {2, 64},
    {3, 32},
    {4, 16}
};

const std::map<unsigned, unsigned> reverse_output_channel_performed_one_shot =
{
    {256, 0},
    {128, 1},
    {64, 2},
    {32, 3},
    {16, 4}
};

const unsigned max_descriptors_x_hw_op = 255;

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

unsigned round_up(unsigned x, unsigned mult)
{
    return ((x + mult - 1) / mult) * mult;
}

unsigned count_bits(unsigned number)
{
    unsigned bits;
    for(bits = 0; number != 0; ++bits)
        number >>= 1;
    return bits;
}

unsigned next_greater_power_of_2(unsigned number)
{
    return pow(2,count_bits(--number));
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

unsigned get_max_mode(ModeSelectionNode node)
{
    return ceil(log2(node.parameters.input_channels)); //At least one channel per ramblock
}

std::vector<unsigned> get_valid_modes(ModeSelectionNode node)
{
    std::vector<unsigned> to_return;
    unsigned max_mode = get_max_mode(node);
    for(unsigned mode = Mode0; mode <= Mode4; ++mode)
    {
        if(mode > max_mode)
            continue;
        unsigned number_of_descriptors_needed = ceil((double)(node.remaining_output_channels) / output_channel_performed_one_shot.at(mode));
        if(number_of_descriptors_needed >= max_descriptors_x_hw_op) //Useless check with the current numbers, but you never know
            continue;
        to_return.push_back(mode);
    }
    return to_return;
}

std::vector<ModeSelectionNode> generateNeighboursComingFromValidModes(ModeSelectionNode current_node)
{
    std::set<ModeSelectionNode> valid_neighbours_set;
    std::vector<unsigned> valid_modes = get_valid_modes(current_node);
    unsigned n = valid_modes.size();
    for(unsigned i = 0; i < n; ++i)
    {
        ModeSelectionNode neighbour(current_node);
        neighbour.remaining_output_channels -= output_channel_performed_one_shot.at(valid_modes[i]);
        if(neighbour.remaining_output_channels < 0)
            neighbour.remaining_output_channels = 0;
        valid_neighbours_set.insert(neighbour);
    }
    return std::vector<ModeSelectionNode>(valid_neighbours_set.begin(), valid_neighbours_set.end());
}

bool check_min_lines_constraint(ConvolutionParameters param)
{
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    //Space required by min lines = min_lines * input_width (rounded up to 16) * input data type size * num input channels
    unsigned space_required = min_lines * round_up(param.input_width, 16) * 2 * param.input_channels;
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

ModeSelectionDistance split_by_input_channel(ConvolutionParameters param, unsigned actual_output_channels, int mode, bool support_split_over_c = true, int split_by_input_channel_overhead = 30000)
{
    ModeSelectionDistance to_return;

    // Min lines  = Kernel_height + kernel stride + 2
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    // maximum ic that I can process without conflicting with min line constraint
    unsigned max_ic_minlines = floor((double)(pow(2,17))/(min_lines * 2 * param.input_width));
    // maximum ic that I can process without conflicting with coefficient line per block constraint
    unsigned max_ic_ramblock = floor((double)(nce1_dpe)/(param.kernel_x*param.kernel_y)*dpe_x_output_channel.at(mode));

    // calculate the max input channels that can be processed without running out of memory:
    unsigned max_ic = std::min(max_ic_minlines, max_ic_ramblock);
    // calculate ramblock for the selected mode
    unsigned ramBlocks = ram_blocks_x_mode.at(mode);
    while (max_ic > 0)
    {
        // max_ic should be divisible by ramblocks and the split should be integer
        if((param.input_channels % max_ic) || (max_ic % ramBlocks))
            max_ic--;
        else
            break;
    }
    if(max_ic == 0)
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = Splits::InputChannel;
        to_return.num_splits = 0;
        to_return.mode = mode;
    }

    // n of input split required (padded to next pow of 2)
    unsigned n_split_c = next_greater_power_of_2(param.input_channels/max_ic);
    unsigned actual_ic_per_split = int(ceil((double)(param.input_channels)/n_split_c));

    if((n_split_c < 2) || (actual_ic_per_split % ramBlocks) || (!support_split_over_c))
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = Splits::InputChannel;
        to_return.num_splits = 0;
        to_return.mode = mode;
    }

    ConvolutionParameters new_param(param);
    param.input_channels = actual_ic_per_split;

    if (check_coefficient_size_constraint(new_param, mode) ||
        check_channels_per_ram_block(new_param, mode)  ||
        check_coefficient_line_constraint(new_param, mode))
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = Splits::InputChannel;
        to_return.num_splits = n_split_c;
        to_return.mode = mode;
    }

    unsigned cost = 0;
    for(unsigned i = 0; i < n_split_c; ++i)
    {
        // calculate the operation cost
        unsigned ic = std::min(actual_ic_per_split, param.input_channels-i*actual_ic_per_split);
        cost += param.kernel_x * param.kernel_y * param.output_width * param.output_height * ic / dpe_x_output_channel.at(mode);
    }
    // add the cost of summation over input channels
    cost += (n_split_c - 1) * (actual_output_channels * param.output_width*param.output_height + split_by_input_channel_overhead);
    to_return.cost = cost;
    to_return.split_performed = Splits::InputChannel;
    to_return.num_splits = n_split_c;
    to_return.mode = mode;
    return to_return;
}

ModeSelectionDistance split_by_width(ConvolutionParameters param, int mode, bool support_split_over_w = true, unsigned split_by_width_overhead = 5000)
{
    ModeSelectionDistance to_return;
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    // Space required by min lines = min_lines * input_width * input data type size * num input channels
    unsigned space_required = min_lines * 2 * param.input_width * param.input_channels;
    unsigned n_split_w = int(ceil(double(space_required)/pow(2, 17)));

    unsigned split_output_width =  param.output_width/n_split_w;
    unsigned split_input_width =  param.input_width/n_split_w;

    if (check_channels_per_ram_block(param, mode) || support_split_over_w)
    {
        to_return.cost = -1;
        to_return.mode = mode;
        to_return.split_performed = Splits::Width;
        to_return.num_splits = n_split_w;
        return to_return;
    }

    unsigned cost = 0;
    for(unsigned i = 0; i < n_split_w; ++i)
    {
        unsigned os_w = std::min(split_output_width, param.output_width-i*split_output_width);
        unsigned is_w = std::min(split_input_width, param.input_width-i*split_input_width);

        ConvolutionParameters new_param(param);
        new_param.output_width = os_w;
        new_param.input_width = is_w;
        if(check_coefficient_size_constraint(new_param, mode) || check_coefficient_line_constraint(new_param, mode))
        {
            to_return.cost = -1;
            to_return.mode = mode;
            to_return.split_performed = Splits::Width;
            to_return.num_splits = n_split_w;
            return to_return;
        }

        cost += param.kernel_x * param.kernel_y * param.output_width * os_w * param.input_channels / dpe_x_output_channel.at(mode) + split_by_width_overhead;
    }
    to_return.cost = cost;
    to_return.num_splits = n_split_w;
    to_return.split_performed = Splits::Width;
    to_return.mode = mode;

    return to_return;
}

ModeSelectionDistance computeModeCost(const ModeSelectionNode a, const ModeSelectionNode b)
{
    ModeSelectionDistance to_return;

    int split_over_output_channel_overhead = 0;
    int mode = 0;
    unsigned output_channel_performed = 0;

    bool need_split_by_k = false;
    if(b.remaining_output_channels == 0)
    {
        output_channel_performed = a.remaining_output_channels;
        std::vector<unsigned> valid_modes = get_valid_modes(a);
        for(mode = valid_modes[valid_modes.size()-1]; mode >= 0; --mode)
        {
            if(output_channel_performed_one_shot.at(mode) >= output_channel_performed)
                break;
        }
        need_split_by_k = false;
        split_over_output_channel_overhead = 0;
    }
    else
    {
        output_channel_performed = a.remaining_output_channels - b.remaining_output_channels;
        mode = reverse_output_channel_performed_one_shot.at(output_channel_performed);
        need_split_by_k = true;
        split_over_output_channel_overhead = 7000;
    }

    ConvolutionParameters parameters = a.parameters;
    int ram_blocks = ram_blocks_x_mode.at(mode);
    parameters.input_channels = round_up(parameters.input_channels, ram_blocks);

    bool need_split_by_width_or_input_channel = check_min_lines_constraint(parameters);
    bool need_split_by_input_channel = check_coefficient_size_constraint(parameters, output_channel_performed) | check_coefficient_line_constraint(parameters, mode);

    if(need_split_by_width_or_input_channel)
    {
        ModeSelectionDistance splitted_by_input_channel = split_by_input_channel(parameters, b.remaining_output_channels, mode);
        ModeSelectionDistance splitted_by_width = split_by_width(parameters, mode);
        if(splitted_by_input_channel < splitted_by_width)
            to_return = splitted_by_input_channel;
        else
            to_return = splitted_by_width;
    }
    else if(need_split_by_input_channel)
        to_return = split_by_input_channel(parameters, b.remaining_output_channels, mode);
    else
    {
        to_return.cost = parameters.kernel_x * parameters.kernel_y * parameters.input_width * parameters.input_height * parameters.input_channels / dpe_x_output_channel.at(mode);
        to_return.split_performed = Splits::NoSplit;
        to_return.num_splits = 1;
        if(check_channels_per_ram_block(parameters, mode))
            to_return.cost = -1; //infinite cost
    }

    to_return.cost += split_over_output_channel_overhead;
    to_return.mode = mode;
    to_return.performed_output_channels = output_channel_performed;
    return to_return;
}
