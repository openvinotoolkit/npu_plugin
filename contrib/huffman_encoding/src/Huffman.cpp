///
/// @file
/// @copyright All code copyright Movidius Ltd 2017, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Huffman encoder/decoder implementation.
///
///

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <map>

#include "Huffman.hpp"
#include "logging.hpp"


using namespace std;

Huffman::Huffman()
{
    reset();
    encodeSymbols = 0;
}

Huffman::Huffman(int nrOfEncodedSymbol)
{
    reset();
    encodeSymbols = nrOfEncodedSymbol;
}

Huffman::~Huffman()
{
    reset();
}

void Huffman::reset()
{
    sumOfBits = sumOfBitsOptimal = originalSize = 0;
    maxLevel = nrOfDistSyms = 0;
    SIZE_OF_SYMBOL = 8;
    RLE_BLOCKS_PADDED_TO_32_BYTES = false;
    RLE_AS_HUFFMAN = true;

    nodes.clear();
    heap = priority_queue<HuffmanTuple_t, vector<HuffmanTuple_t>,
            greater<HuffmanTuple_t>>();
    codedSyms.clear();
    levelLeavesCnt.clear();
    encSyms.clear();
    encSymLengths.clear();
    interiorNodes.clear();
    symAddr.clear();
    inbuf_size.clear();
    buff_bit_count.clear();
    pipe_padding.clear();
}

void Huffman::constructHeap(const vector<Symbol> &data, int bpb)
{
    HuffmanTuple_t t;
    for (unsigned int i = 0; i < data.size(); i++)
    {
        t = HuffmanTuple_t(data[i].symbol, data[i].occurrences, i, -1, -1);
        nodes.push_back(t);
    }
    for (HuffmanTuple_t &it : nodes)
        heap.push(it);
    SIZE_OF_SYMBOL = bpb;
    nrOfDistSyms = nodes.size();
    if(encodeSymbols > 0)
    {
        while((int) heap.size() > encodeSymbols)
            heap.pop();
    }
}

/* Function to reverse bits of num */
unsigned short int reverseBits_short(unsigned short int num)
{
    unsigned short int  NO_OF_BITS = sizeof(num) * 8;
    unsigned short int reverse_num = 0, i, temp;
 
    for (i = 0; i < NO_OF_BITS; i++)
    {
        temp = (num & (1 << i));
        if(temp)
            reverse_num |= (1 << ((NO_OF_BITS - 1) - i));
    }
  
    return reverse_num;
} 

HuffmanResult_t Huffman::encode(const Symbol *data, int len, int bpb, bool bypass)
{
    return encode(vector<Symbol>(data, data + len), bpb, bypass);
}

HuffmanResult_t Huffman::encode(const vector<Symbol> &data, int bpb, bool bypass)
{
    HuffmanTuple_t a, b, c;
    stringstream rptStream;

//    unsigned char *byte = (unsigned char *) data;

    reset();
    constructHeap(data, bpb);

    for (HuffmanTuple_t &it : nodes) {
        originalSize += it.occurrences * SIZE_OF_SYMBOL;
//        Log(5, "originalSize %lld ", originalSize);
    }

    sumOfBits = 0;
    sumOfBitsOptimal = 0;
    codedSyms.clear();
    maxLevel = 0;
    mode = 2;

    if (bypass) {
        mode = 1;
        return HuffmanResult_t(originalSize, sumOfBits, sumOfBitsOptimal);
    }

    int leavesOccurrences = 0;
    //---------------------------------------------------------------------------------------------------------------------
    // RLE mode: Entered if Huffman table has only 1 symbol, (RLE compression)
    //---------------------------------------------------------------------------------------------------------------------
    if(heap.size() == 1)
    {
        if (RLE_AS_HUFFMAN) {
            while (nodes.size() < 16) {
                a = heap.top();
                Log(1, "encode: A node: %s, index: %0d, occurrences: %0d", a.getFullSymbol().c_str(), a.index, a.occurrences);
                nodes.push_back( b = HuffmanTuple_t(std::string(1,0), 0, nodes.size(), -1, -1) );
                heap.push(b);
            }
            while (heap.size() > 1) {
                a = heap.top();
                heap.pop();
                b = heap.top();
                heap.pop();
                Log(1, "encode: A node: %s, index %0d B node: %s, index %0d", a.getFullSymbol().c_str(), a.index, b.getFullSymbol().c_str(), b.index);

                c = HuffmanTuple_t(a.symbol + b.symbol, a.occurrences + b.occurrences, nodes.size(), a.index, b.index);
                nodes.push_back(c);
                heap.push(nodes.back());
            }
        } else {
            Log(1, "encode: RLE MODE");
            mode = 2;

            dataSize = nodes[0].occurrences;    // Data size = symbol occurrences
            sumOfBits = 32;                     // total bits = 32 bit METADATA (no data required)
            sumOfBitsOptimal = sumOfBits;

            // @@ dccormac -- edited "METADATA" for RLE mode 
            // bit  [0]     BYPASS flag
            // bit  [1]     RLE flag
            // bits [13:2]  Num of symbol repetition == Data size -1 in bytes
            // bits [21:14] RLE symbol
            // bits [31:22] Padded with zeros

            headerBits = ((nodes[0].symbol[0] & 0x000000FF) << 14) | (((dataSize-1) & 0x00000FFF) << 2) | 0x2;

            Log(3, "encode: RLE headerBits 32 bit value: 0x%x  \n        bit [0] bypass mode: 0x%x \n        bit [1] RLE mode: 0x%x \n        bits [13:2] size of data: 0x%x \n        bits [21:14] symbol: 0x%x \n        bits [32:22] padded: 0x%x \n", headerBits, headerBits & 0x00000001, (headerBits & 0x00000002) >> 1, (headerBits & 0x00003FFC) >> 2, (headerBits & 0x003FC000) >> 14, (headerBits & 0x3C00000) >> 22);

            return HuffmanResult_t(originalSize, sumOfBits, sumOfBitsOptimal);
        }
    } else {
        //---------------------------------------------------------------------------------------------------------------------
        // HUFFMAN or BYPASS mode depending on the compression rate
        //---------------------------------------------------------------------------------------------------------------------
        while (heap.size() > 1)
        {
            do
            {
                a = heap.top();
                heap.pop();
            } while (a.occurrences == 0);
            b = heap.top();
            heap.pop();

            // Partial encoding only
            if(encodeSymbols != 0)
            {
                Log(5, "encode: A node: %s, index %0d B node: %s, index %0d", a.getFullSymbol().c_str(), a.index, b.getFullSymbol().c_str(), b.index);
                if(a.leftSon == -1) {
                    leavesOccurrences += a.occurrences;      
                    Log(5, "leavesOccurrences %d ",leavesOccurrences);
                    Log(5, "a.occurrences %d ",a.occurrences);
                }

                if(b.leftSon == -1) {
                    leavesOccurrences += b.occurrences;
                    Log(5, "leavesOccurrences %d ",leavesOccurrences);
                    Log(5, "b.occurrences %d ",b.occurrences);
                }
            }

            c = HuffmanTuple_t(a.symbol + b.symbol, a.occurrences + b.occurrences,
                    nodes.size(), a.index, b.index);
            nodes.push_back(c);
            heap.push(nodes.back());
        }
    }

    DFS(heap.top(), 0, 0);      // Computing each encoding for all leaves
    generateEncodedSymbols();

    //----------------------------------------------------------------------------------------------------------------------------
    // below is where the inbuf details are added 
    //----------------------------------------------------------------------------------------------------------------------------
    // each input buffer size is calculated in terms of number of symbols per buffer.

    int DS_B;               // data size in Bytes
    inbuf_size.resize(13);  // number of input buffers (pipes)
    int temp =0;

    DS_B = originalSize /  SIZE_OF_SYMBOL; 
    Log(3, "encode: originalSize in BYTES: %d ", DS_B);

    // distribute the symbols across the pipe 
    // for 4kB blocks
    if(DS_B == 4096) {     
        for (int j=0; j<12; j++) {
            inbuf_size[j] = 316;
        }
        inbuf_size[12] = 304;
    }
    // for blocks less then 4kB
    // Less than a full pipe, Put everything into the first pipe
    if (DS_B < 128) {
        inbuf_size[0] = DS_B;
        for (int i=1; i<13; i++) {
            inbuf_size[i] = 0;
        }
    }
    //   1664 = 128 per pipe (the min number of symbols there can be in a pipe)
    else if (DS_B < 1664) {              
        int num_pipes = DS_B / 128;
        int symbols_per_pipe = DS_B / num_pipes;
        for (int i=0; i<num_pipes-1; i++) {
            inbuf_size[i] = symbols_per_pipe;
        }
        inbuf_size[num_pipes-1] = DS_B - (symbols_per_pipe * (num_pipes-1));
        for (int i=num_pipes; i<13; i++) {
            inbuf_size[i] = 0;
        }
        Log(5, "Data spread evenly across %0d pipes", num_pipes);
    }
    // distribute across all 13 pipe as evenly as possible
    else {
        temp = ceil(DS_B / 13.0);
        for (int i=0; i<12; i++) {
            inbuf_size[i] = temp;
        }
        inbuf_size[12] = DS_B - (temp*12);
    }

    Log(5, "encode: inbuf_size.size() %lu \n", inbuf_size.size());
    for (size_t i = 0; i < inbuf_size.size(); i++) {
        Log(5, " inbuf size_%lu size = %d", i, inbuf_size[i]); 
    }

    //----------------------------------------------------------------------------------------------------------------------------

    mode = 1; // Huffman or Bypass mode bepending on compression size
 
    sumOfBits += 432; // Huffman metadata + data size in bits (not including padding required at the end of each pipe, calculated the writing stage)

    // Overhead for partial encoding only
    if(encodeSymbols != 0)
    {
        sumOfBits += (originalSize - leavesOccurrences * SIZE_OF_SYMBOL); // Uncompressed bits
        Log(5, "encode: HUFFMAN sumOfBits: %lld \n", sumOfBits);
        sumOfBits += originalSize / SIZE_OF_SYMBOL; // Bits for encoded or not
        Log(5, "encode: partial encoding ");
        Log(5, "encode: originalSize: %lld ", originalSize);
        Log(5, "encode: Total sumOfBits (excluding pipe '0' padding) :  %lld ", sumOfBits);
    }

    // Data size (excluding pipe '0' padding) to be written into file (padding calculated in the write to file function)
    dataSize = sumOfBits - 432;  // 432 == size of metadata in bits

    Log(3, "encode: originalSize: %lld ", originalSize);
    Log(3, "encode: encoded data Size  (excluding pipe '0' padding) %d ", dataSize);
    Log(5, "encode: Max level of tree: %d", maxLevel);
                       // total size, compressed size, entropy size 
    return HuffmanResult_t(originalSize, sumOfBits, sumOfBitsOptimal);
}

void Huffman::DFS(const HuffmanTuple_t &node, int steps, uint32_t code)
{
    Log(4, "DFS: evaluating step %0d - Node symbol 0x%s", steps, node.getFullSymbol().c_str());
    if (node.leftSon == -1)
    {
        sumOfBits += steps * node.occurrences;
        sumOfBitsOptimal = sumOfBitsOptimal
                + log2(1.0 * originalSize / SIZE_OF_SYMBOL / node.occurrences)
                        * node.occurrences;
        if(maxLevel < steps)
        {
            maxLevel = steps;
            levelLeavesCnt.resize(maxLevel + 1);
        }
        levelLeavesCnt[steps]++;
        encSymLengths.push_back(make_pair(steps, node.symbol));
        Log(4, "DFS: Tree step %0d - Node symbol 0x%0x added", steps, (int) (node.symbol[0]) & 0xFF);
        return;
    }
    DFS(nodes[node.leftSon], steps + 1, (code << 1));
    DFS(nodes[node.rightSon], steps + 1, ((code << 1) + 1));
}

void Huffman::generateEncodedSymbols()
{
    sort(encSymLengths.begin(), encSymLengths.end());
    encSyms.push_back((1 << encSymLengths[0].first) - 1);
    codedSyms.insert(make_pair(encSymLengths[0].second,
                               HuffmanCoded_t(encSyms[0],
                                       encSymLengths[0].first)));

    Log(4, "generateEncodedSymbols: encoded Symbol %0x, length: %0d from original byte 0x%0x", encSyms[0], encSymLengths[0].first, encSymLengths[0].second[0] & 0xFF);

    for(size_t i = 1; i < encSymLengths.size(); i++) {
        auto &symbol = encSymLengths[i].second;
        const int sh = (encSymLengths[i].first - encSymLengths[i - 1].first);

        encSyms.push_back((encSyms[i - 1] << sh) - 1);
        codedSyms.insert(make_pair(symbol, HuffmanCoded_t(encSyms[i], encSymLengths[i].first)));

        Log(4, "generateEncodedSymbols: encoded Symbol %0x, length: %0d from original byte 0x%0x", encSyms[i], encSymLengths[i].first, symbol[0] & 0xFF);
    }

    interiorNodes.push_back(1);
    Log(4, "Root node added");
    for (int k = 1; k <= maxLevel; k++) {
        interiorNodes.push_back(interiorNodes[k - 1] * 2 - levelLeavesCnt[k]);
        Log(4, "%0d interior nodes added to level %0d", interiorNodes[k], k);
    }

    for(size_t i = 0; i < encSyms.size(); i++)  {
        int val = encSyms[i] - interiorNodes[encSymLengths[i].first];
        for(int k = 2; k <= encSymLengths[i].first; k++)
            val += levelLeavesCnt[k - 1];
        symAddr.push_back(val);
        Log(4, "generateEncodedSymbols: new encoded value %0x and its corresponding pre-encoded value 0x%0x", val, (int) (encSymLengths[i].second[0]) & 0xFF);
    }
}

vector<Symbol> Huffman::getSymFreqs(const void *data, int length, int bits)
{
    vector<Symbol> freq;
    map<string, int> freqDict;
    unsigned char *it = (unsigned char*) data;
    int bytes = (bits / 8) + (bits % 8 != 0);

    for (int i = 0; i < length; i++) {
        string x;
        for (int j = 0; j < bytes; j++, i++) {
            if (i < length)
                x += it[i];
            else
                x += '\0';
        }
        i--;
        freqDict[x]++;
    }
    for (auto &kv : freqDict) {
        freq.push_back(Symbol(kv.first, kv.second));
        Log(4, "getSymFreqs: symbol 0x%02x with frequency %0d added", kv.first[0] & 0xFF, kv.second);
    }

    return freq;
}

HuffmanCoded_t Huffman::getSymbolCode(char sym)
{
    return getSymbolCode(string() + sym);
}

HuffmanCoded_t Huffman::getSymbolCode(const string &sym)
{
    auto it = codedSyms.find(sym);
    if (it != codedSyms.end())
        return it->second;
    return HuffmanCoded_t();
}

static int writeToFile(const char *encodedValues, int len, const string &filename)
{
    FILE *fout = fopen(filename.c_str(), "ab");
    if (!fout) {
        throw Exception(
            PrintToString(
                "Could not open file %s", filename.c_str()));
   }

//    if (!fout) {
//        cerr << "Could not open file '" << filename.c_str() << "'!\n";
//        return -1;
//    }
//    cout << " to file '" << filename.c_str() << "'\n";
    Log(5, "writeToFile: writing %x \n", *encodedValues);

    int n = fwrite(encodedValues, 1, len, fout);
    fclose(fout);
    assert(n == len);

    return 0;
}


static int writeToBuffer(const char *encodedValues, int len, vector<char> *outputBuffer)
{
    uint32_t lviInitialLength;
    uint32_t lviDeltaLength;
    stringstream rptStream;
    
    lviInitialLength = outputBuffer->size();
    lviDeltaLength   = static_cast<uint32_t>(len);
    
    rptStream << "writeToBuffer: outputBuffer.size(): " << outputBuffer->size() << " bytes; adding " << lviDeltaLength << " bytes" << endl;
    Report (3, rptStream);
    
    // do a resize here
    
    for ( int i = 0; i < len; i++ )
    {
        outputBuffer->push_back(encodedValues[i]);
    }

    if ( outputBuffer->size() == ( lviInitialLength + lviDeltaLength ) )
    {
        rptStream << "writeToBuffer: done (outputBuffer->size(): " << outputBuffer->size() << ")" << endl;
        Report (3, rptStream);
        return 1;
    }
    else
    {
        rptStream << "writeToBuffer: failed (outputBuffer->size(): " << outputBuffer->size() << ")" << endl;
        Error (0, rptStream);
        return 1;
    }

}

int Huffman::writeEncodedDataToFile( const void *data,
                                              int length,
                                              const std::string &filename,
                                              bool bypass,
                                              bool statsOnly
                                              )
{
    huffmanOutputDataRouting_t lvtHuffmanOutputDataRouting;
    vector<char> *dummyDataBuffer = NULL;
    
    lvtHuffmanOutputDataRouting = WRITE_TO_FILE;
    
    return writeEncodedData( data,
                             length,
                             filename,
                             dummyDataBuffer,
                             bypass,
                             statsOnly,
                             lvtHuffmanOutputDataRouting
                             );

}

int Huffman::writeEncodedData( const void *data,
                               int length,
                               const std::string &outputDataFileName,
                               std::vector<char> *outputDataBuffer,
                               bool bypass,
                               bool statsOnly,
                               huffmanOutputDataRouting_t &outputDataRouting
                               )
{
    Log(5, "writeEncodedData: In writeEncodedDataToFile\n");
    BitData encodedValues;
    encodedValues.length = 0;
    bool buffer_dist_failed = false;
    stringstream rptStream;

    //---------------------------------------------------------------------------------------------------------------------
    // RLE mode
    //---------------------------------------------------------------------------------------------------------------------
    // Just write 32-bit of metadata (no data required)
    if(mode == 2 && !RLE_AS_HUFFMAN) {
        Log(3, "writeEncodedData: RLE MODE");

        //------------------------------------
        // created RLE header data (metadata)
        //------------------------------------
        // bit  [0]     BYPASS flag
        // bit  [1]     RLE flag
        // bits [13:2]  Num of symbol repetition == Data size -1 in bytes
        // bits [21:14] RLE symbol
        // bits [31:22] Padded with zeros

        encodedValues.addInt(headerBits, 32);  // number of bits to be inserted for Metadata
        
        if (!statsOnly) {
            int status = 0;
            if ( outputDataRouting == WRITE_TO_FILE )
            {
                if((status = writeToFile(encodedValues.bits.data(), encodedValues.bits.size(), outputDataFileName))) {
                    return status;
                }
            }
            else
            {
                if((status = writeToBuffer(encodedValues.bits.data(), encodedValues.bits.size(), outputDataBuffer))) {
                    return (encodedValues.bits.size()*8);
                }
            }
        }
        return (32);
    }

    //---------------------------------------------------------------------------------------------------------------------
    // Compression is used, HUFFMAN mode
    //---------------------------------------------------------------------------------------------------------------------
    // Write METADATA and symbols to file 
//    if(mode == 1) {
    else {
        // ------------------------------------------------------------------------------------------------------------------
        // generating details for INPUT BUFF sizes 0 - 12 
        // calculating the number of bit per pipe (including start bits but NOT including any padding)
        unsigned char *byte = (unsigned char *) data;
        int count = 0;
        int byte_count = 0;   
        buff_bit_count.resize(13);

        // Keep trying to solve the buffer distribution until all pipes are valid
        bool all_valid = false;
        bool buf_filled [13] = {};
        while (!all_valid && !buffer_dist_failed) {
            byte_count = 0;
            for (unsigned int j=0; j < inbuf_size.size(); j++) {
                count = 0;
                int num_syms_in_window = 0;//number of symbols seen in a 24 bit wide sliding window so far
                int flipped[4];
                int num_max_syms = 4; // Tracks the number of symbols in the 24bit window.
                int start_next_window_flipped = 0;
                for (int k=0; k < inbuf_size[j]; k ++) {
                    HuffmanCoded_t coded = getSymbolCode(byte[byte_count]);
                    //Start New Check 
                    if (num_syms_in_window == 0) {
                        for(int init=0;init<4;init++) {
                            flipped[init] = 0;
                        }
                        int origNumofBits[4];
                        // Get original code lengths for the next 3 symbols.
                        for(int lpidx=0;lpidx<4;lpidx++) {
                            HuffmanCoded_t code  = getSymbolCode(byte[byte_count+lpidx]);
                            origNumofBits[lpidx] = 1 + (code.nrOfBits ? code.nrOfBits : SIZE_OF_SYMBOL);
                        }
                        // Each of teh 3 loops below track the flipped status of the symbols[0],[1][2]
                        // flipped means un-encoded irrespective of if it was already un-encoded or not
                        for(int fli=start_next_window_flipped;fli < 2; fli++) {
                            int sumi = fli ? (1+SIZE_OF_SYMBOL) : origNumofBits[0];
                            // No Check here since the first symbol alone cannot exceed 24 or be equal to 24.
                            for(int flj=0;flj < 2; flj++) {
                                int sumj = flj ? (1+SIZE_OF_SYMBOL) : origNumofBits[1];
                                if ( (sumi + sumj == 24) || (sumi + sumj == 25) ){ 
                                    // Continue and check if flipping the second symbol to non-encoded helps.
                                    continue ;
                                } else if (sumi + sumj > 25) {
                                    flipped[0] = fli;
                                    num_max_syms = 1;
                                    if (flj) {
                                        start_next_window_flipped = 1;
                                        //printf("Start symbol of next window flipped\n");
                                    } else {
                                        start_next_window_flipped = 0;
                                    }
                                    goto found_solution;
                                }
                                for(int flk=0;flk < 2; flk++) {
                                    int sumk = flk ? (1+SIZE_OF_SYMBOL) : origNumofBits[2];
                                    if ( (sumi + sumj + sumk  == 24) || (sumi + sumj + sumk  == 25) ) { 
                                        // Continue and check if flipping the third symbol to non-encoded helps.
                                        continue ;
                                    } else  if (sumi + sumj + sumk  > 25) {
                                        flipped[0] = fli;
                                        flipped[1] = flj;
                                        num_max_syms = 2;
                                        if (flk) {
                                            start_next_window_flipped = 1;
                                            //printf("Start symbol of next window flipped\n");
                                        } else {
                                            start_next_window_flipped = 0;
                                        }
                                        goto found_solution;
                                    } else {
                                        // Sum of first 3 is less than 24, check if the 4 would cross the 
                                        // 24 bit boundary. If no, the max sym should be 4.
                                        flipped[0] = fli;
                                        flipped[1] = flj;
                                        flipped[2] = flk;
                                        if (sumi + sumj + sumk + origNumofBits[3] <= 25) {
                                            num_max_syms = 4;
                                        } else {
                                            num_max_syms = 3;
                                        }
                                        start_next_window_flipped = 0;
                                        goto found_solution;
                                    }
                                }
                            }
                        }
                        printf("No solution found\n");
                        assert(0);
found_solution: ;
                    }
                                

                    //if(k != inbuf_size[j]-1 || ((j != inbuf_size.size() - 1) && inbuf_size[j+1] != 0))
                    //{// this means we should have one extra lookahead byte/symbol available
                        //assert((byte_count + 1) < length);// let's double-check to be sure

                        if (flipped[num_syms_in_window] && 1) 
                        {
                            coded.nrOfBits = 0;
                        }
                        num_syms_in_window++;
                        if(num_syms_in_window == num_max_syms)
                        {
                            num_syms_in_window = 0;
                        }
                        assert(num_syms_in_window < 4);
                    //}
                    // If the number of bits in the buffer will be more than the maximum size of the buffer
                    if (count + 1 + (coded.nrOfBits ? coded.nrOfBits : SIZE_OF_SYMBOL) > MAX_BUF_BIT_SIZE) {
                        Log(1, "writeEncodedData: Number of symbols in buffer will overflow maximum buffer size. Number of symbols: %0d - Bit length: %0d", inbuf_size[j], count + 1 + (coded.nrOfBits ? coded.nrOfBits : SIZE_OF_SYMBOL));
                        // If all previous buffers are filled, set this one to filled
                        if (j == 0) { buf_filled[j] = true; }
                        for (unsigned int buf = 0; buf < j; ++buf) {
                            if (!buf_filled[buf]) break;
                            if (buf == j-1) {
                                buf_filled[j] = true;
                                Log(1, "writeEncodedData: All previous buffers already filled. Setting filled also");
                            }
                        }
                        // Redistribute the remaining symbols among the other buffers that can hold more.
                        // First need to know how many are unfilled and to be used.
                        int unfilled_buffers = 0;
                        for (unsigned int buf = 0; buf < inbuf_size.size(); ++buf) {
                            if (!buf_filled[buf] && inbuf_size[buf] != 0 && buf != j) {
                                // Clear the buffer filled status of any succesive buffers. Symbols will be changed.
                                for (unsigned int temp_buf = buf+1; temp_buf < inbuf_size.size(); ++temp_buf) {
                                    buf_filled[temp_buf] = false;
                                }
                                ++unfilled_buffers;
                            }
                        }
                        if (unfilled_buffers == 0) {
                            Log(1, "writeEncodedData: All buffers already filled to capacity. Switching to bypass");
                            buffer_dist_failed = true;
                            goto buffer_dist_loop_end;
                        }
                        // Divide the remaining symbols among the other buffers
                        int spilled_symbols = (inbuf_size[j] - (k - 1)) / unfilled_buffers;
                        int spilled_remainder = (inbuf_size[j] - (k - 1)) % unfilled_buffers;
                        Log(1, "writeEncodedData: Dividing %0d symbols among %0d buffers", inbuf_size[j] - (k - 1), unfilled_buffers);
                        for (unsigned int buf = 0; buf < inbuf_size.size(); ++buf) {
                            if (!buf_filled[buf] && inbuf_size[buf] != 0 && buf != j) {
                                Log(1, "writeEncodedData: Buffer %0d gets %0d more symbols", buf, spilled_symbols + spilled_remainder != 0);
                                inbuf_size[buf] += spilled_symbols;
                                // Also add any remainder of the division, one per buffer until all distributed.
                                if (spilled_remainder) {
                                    ++inbuf_size[buf];
                                    --spilled_remainder;
                                }
                            }
                        }
                        Log (1, "writeEncodedData: Buffer %0d previously had %0d symbols. Now has %0d", j, inbuf_size[j], k - 1);
                        inbuf_size[j] = k - 1;
                        goto buffer_dist_loop_end;
                    }
    
                    byte_count++;
                    count ++;                               // add 1 for start bit
                    if(coded.nrOfBits != 0) {               // symbol was encoded
                        count = count + coded.nrOfBits;     // add to num of bits used to encode the symbol
                        Log(5, "writeEncodedData: Symbol Encoded ");
                        Log(5, "count = %d  J = %d   k = %d ", count, j, k); 
                    }
                    else {
                        count = count + SIZE_OF_SYMBOL;     // add 8 bit (SIZE_OF_SYMBOL)
                        Log(5, "count = %d  J = %d   k = %d ", count, j, k); 
                    }
                }
                buff_bit_count[j] = count;
                Log(5, "writeEncodedData: count = %d j = %d  buff_bit_count[%d] = %d ", count, j, j, buff_bit_count[j]); 
                Log(5, "writeEncodedData: bit count %d ", byte_count);
                Log(5, "writeEncodedData: originalSize %lld ", originalSize);
            }
            all_valid = true;
buffer_dist_loop_end: ;
        }

        Log(5, "writeEncodedData: byte count %d ", byte_count);
        Log(5, "writeEncodedData: originalSize %lld ", originalSize);
//        assert(byte_count == originalSize / SIZE_OF_SYMBOL); // origional size in bits/8 (bytes), byte_count = total no of Symbols 
        assert(length == originalSize / SIZE_OF_SYMBOL);     // length passed from main

        // ------------------------------------------------------------------------------------------------------------------
        // calculate the Overall size of the ENCODED Data (encoded block size - 1 , including start bits and 0 padding, in bytes)
        pipe_padding.resize(13);
        int encoded_data_size = 0;      // encoded data size (excluding metadata)
        int t_encoded_data_size = 0;    // including padding

        for (unsigned int i=0; i < buff_bit_count.size(); i++) {
            Log(5, "writeEncodedData: pipe_padding remainder %d ", buff_bit_count[i] % 16);
            if (buff_bit_count[i] % 16 != 0) {
                pipe_padding[i] = 16 - (buff_bit_count[i] % 16);
                Log(5,"writeEncodedData: pipe_padding[%d] = %d \n", i, pipe_padding[i]);
            }
            encoded_data_size += buff_bit_count[i];
            t_encoded_data_size += buff_bit_count[i] + pipe_padding[i];  // adding '0' padding
            Log(5, "writeEncodedData: buff_bit_count[%d] = %d  pipe_padding[%d] = %d  encoded data size = %d ", i, buff_bit_count[i], i, pipe_padding[i], t_encoded_data_size);
        }

        Log(3, "writeEncodedData: Total encoded data size (excluding padding & metadata) = %d ", encoded_data_size);
        Log(3, "writeEncodedData: Total encoded data size including '0' padding (excluding metadata) = %d ", t_encoded_data_size);

        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        // BYPASS MODE: Entered If HUFFMAN compression is worse than origional 
        // or bypass mode selected
        // or compressed data size (including padding) is 50% or less then the original data size 
        if (((t_encoded_data_size + 432) > (originalSize + 16)) | bypass | buffer_dist_failed) {
            if (bypass) {
                rptStream << "writeEncodedData: BYPASS MODE selected on command line" << endl;
                Report(1, rptStream);
                Log(1, "writeEncodedData: BYPASS MODE selected on command line ");
            }
            else if (buffer_dist_failed) {
                rptStream << "writeEncodedData: Couldn't allocate symbols into buffers. Switching to bypass" << endl;
                Report(1, rptStream);
                Log(1, "writeEncodedData: Couldn't allocate symbols into buffers. Switching to bypass");
            }
            else {
                rptStream << "writeEncodedData: Encoded data size, including Metadata (" << (t_encoded_data_size + 432) << ") is greater than original data size (" << (originalSize + 16) << ")" << endl;
                rptStream << "writeEncodedData: Changing to BYPASS MODE" << endl;
                Report(1, rptStream);
                Log(1, "writeEncodedData: Encoded data size, including Metadata (%d) is greater than original data size (%lld) ",t_encoded_data_size + 432, originalSize + 16 );
                Log(1, "writeEncodedData: Changing to BYPASS MODE ");
            }
        //------------------------------------
        // created BYPASS header data (metadata)
        //------------------------------------
            // bit  [0]     BYPASS flag
            // bit  [1]     RLE flag
            // bits [13:2]  Data size -1 in bytes
            // bits [15:14] Padded with zeros

            dataSize = originalSize / SIZE_OF_SYMBOL;  // in bytes
            // The data size must bea  multiple of 2 bytes in bypass mode. Total data should be padded to a 16 bit boundry.
            if (dataSize % 2) {
                ++dataSize;
            }
            Log(5, "writeEncodedData: data size %x ", dataSize);
            sumOfBits = originalSize + 16;  // 16 bits for matadata 
            if (sumOfBits % 16) {
                sumOfBits += 16 -  (sumOfBits % 16);
            }
            Log(5, "writeEncodedData: sumOfBits size %llx ", sumOfBits);    

            headerBits = ((dataSize-1) << 2) | 0x0001;
            Log(3, "writeEncodedData: BYPASS headerBits 16 bit value: 0x%x  \n        bit [0] bypass mode: 0x%x  \n        bit [1] RLE mode: 0x%x \n        bits [13:2] size of data: 0x%x \n        bits [15:14] padded 0x%x \n", headerBits, headerBits & 0x0001, (headerBits & 0x0002) >> 1, (headerBits & 0x3FFC) >> 2, (headerBits & 0xC0000) >> 14);

            // Just write 16-bit of metadata + copy all original symbols 
            unsigned char *Bytes = (unsigned char*) data;
            int mask = ((1 << SIZE_OF_SYMBOL) - 1);     // create a MASH for the non encoded symbols (= FF as SIZE_OF_SYMBOL == 8)
            encodedValues.addInt(headerBits, 16);  // number of bits to be inserted for Metadata
            for (int i=0; i<length; ++i) {
                encodedValues.addInt(Bytes[i] & mask, SIZE_OF_SYMBOL);
            }
            // Add extra padding to data.
            for (int unsigned i=length; i<dataSize; ++i) {
                encodedValues.addInt(0 & mask, SIZE_OF_SYMBOL);
            }
            if (!statsOnly) {
                int status = 0;
                Log(5, "writeEncodedData: writing Metadata ");
                if ( outputDataRouting == WRITE_TO_FILE )
                {
                    if((status = writeToFile(encodedValues.bits.data(), encodedValues.bits.size(), outputDataFileName)))
                    {
                        return encodedValues.bits.size()*8;
                    }
                }
                else
                {
                    if((status = writeToBuffer(encodedValues.bits.data(), encodedValues.bits.size(), outputDataBuffer)))
                    {
                        return encodedValues.bits.size()*8;
                    }
                }
            }
            return (sumOfBits);
        } // end of Bypass mode

        rptStream << "writeEncodedData: HUFFMAN MODE" << endl;

        Report(1, rptStream);
        Log(1, "writeEncodedData: HUFFMAN MODE\n");

        Log(3, "writeEncodedData: HUFFMAN Metadata 432 bits ");

        PrintSymbolTable();
        PrintLeavesTable();
    //------------------------------------
    // created HUFF header data (metadata)
    //------------------------------------
        // bit  [0]       BYPASS flag
        // bit  [1]       RLE flag
        // bits [13:2]    Data size -1 in bytes     // encoded data block in bytes including the ZERO padding at the end of each pipe
        // bits [133:14]  Skip Table
        // bits [261:134] Symbol table
        // bits [273:262] input buffer size 0       // size in bits of each pipe (does not include ZERO pading at end of pipe)
        // bits [285:274] input buffer size 1
        // bits [297:286] input buffer size 2
        // bits [309:298] input buffer size 3
        // bits [321:310] input buffer size 4
        // bits [333:322] input buffer size 5
        // bits [345:334] input buffer size 6
        // bits [357:346] input buffer size 7
        // bits [369:358] input buffer size 8
        // bits [381:370] input buffer size 9
        // bits [393:382] input buffer size 10
        // bits [405:394] input buffer size 11
        // bits [417:406] input buffer size 12
        

        // ------------------------------------------------------------------------------------------------------------------
        // Add Mode the length in bit 
        encodedValues.addInt(0, 2);  // 0x0 2 bits indicating Huffman mode
        Log(3, "   2 bits for mode indicating Huffman ");

        // ------------------------------------------------------------------------------------------------------------------
        // Add data size 
        t_encoded_data_size = (t_encoded_data_size / SIZE_OF_SYMBOL) -1;
        Log(3, "writeEncodedData: HUFFMAN Total encoded data size including '0' padding (excluding metadata) = %d ", t_encoded_data_size);
        encodedValues.addInt(t_encoded_data_size, 12);  

        // ------------------------------------------------------------------------------------------------------------------
        // Add SKIP TABLE (leaf count at each level
        for (int i = 1; i < 16; i++)  // 15 = max encoding symbol length 
        {
            Log(5, "writeEncodedData: HUFFMAN skip table i = %d  ", i);
            Log(5, "maxLevel = %d ", maxLevel);
            if(i < (int) levelLeavesCnt.size()) {
                encodedValues.addInt(levelLeavesCnt[i], 8); // 8 bits per value
                Log(3, "   skip table: %d ", levelLeavesCnt[i]);
            }
            else {
                encodedValues.addInt(0, 8);
                Log(3, "   skip table: 0 ");
            }
        }

        // ------------------------------------------------------------------------------------------------------------------
        // Add original symbols being encoded  
        for (size_t i = 0; i < 16; i++)                    
        {
            if (i < symAddr.size()) {
                encodedValues.addInt(encSymLengths[symAddr[i]].second[0], 8);
                Log(3, "   symbols encoded: 0x%02x ", encSymLengths[symAddr[i]].second[0] & 0xFF);
            }
            else {
                encodedValues.addInt(0, 8);
                Log(3, "   symbols encoded: 0x00 ");
            }
        }

        // ------------------------------------------------------------------------------------------------------------------
        // Add input buffers values (including start bits but NOT including any padding)
        for (size_t i = 0; i < inbuf_size.size(); i++)                    
        {
            encodedValues.addInt(buff_bit_count[i], 12);
            Log(3, "   in_buff_size_%lu: %d", i, buff_bit_count[i]);
        }

        // ------------------------------------------------------------------------------------------------------------------
        // Add padding to Metadata (14 bits)
            encodedValues.addInt(0, 14);
             Log(4, "   14 bits of padding ");

        // ------------------------------------------------------------------------------------------------------------------
        // add encoded and non-encoded symbols & add the required padding at the end of each pipe 
        unsigned char *Bytes = (unsigned char*) data;
        int mask = ((1 << SIZE_OF_SYMBOL) - 1);     // create a MASH for the non encoded symbols (= FF as SIZE_OF_SYMBOL == 8)

         Log(5, "\nwriteEncodedData: HUFFMAN SYMBOLS ");
        int byte_o = 0, incrBits = 0;
        for (unsigned int j=0; j < inbuf_size.size(); j++) {                // loop through each pipe (13)
            int num_syms_in_window = 0;//number of symbols seen in a 24 bit wide sliding window so far
            int flipped[4];
            int num_max_syms = 4;
            int start_next_window_flipped = 0;
            int num_bits_in_window = 0;
            int window_number_in_pipe = 0;
            for (int k=0; k < inbuf_size[j]; k ++) {                        // loop inside the size of each tile (each symbol)
                HuffmanCoded_t code = getSymbolCode(Bytes[byte_o]);
                //Start New Check 
                if (num_syms_in_window == 0) {
                // At the start of the 25 bit window, initialized all the flipped status of the 4 
                // possible symbols to 0. Indicating that we are leaving the symbol as is and not deciding to
                // leave it unencoded.
                    for(int init=0;init<4;init++) {
                        flipped[init] = 0;
                    }
                    int origNumofBits[4];
                    // Get original code lengths for the next 3 symbols.
                    for(int lpidx=0;lpidx<4;lpidx++) {
                        HuffmanCoded_t coded  = getSymbolCode(Bytes[byte_o+lpidx]);
                        origNumofBits[lpidx] = 1 + (coded.nrOfBits ? coded.nrOfBits : SIZE_OF_SYMBOL);
                    }
                    // Each of teh 3 loops below track the flipped status of the symbols[0],[1][2]
                    // flipped means un-encoded irrespective of if it was already un-encoded or not
                    for(int fli=start_next_window_flipped;fli < 2; fli++) {
                        int sumi = fli ? (1+SIZE_OF_SYMBOL) : origNumofBits[0];
                        // No Check here since the first symbol alone cannot exceed 24 or be equal to 24.
                        for(int flj=0;flj < 2; flj++) {
                            int sumj = flj ? (1+SIZE_OF_SYMBOL) : origNumofBits[1];
                            if ( (sumi + sumj == 24) || (sumi + sumj == 25) ) { 
                                // Continue and check if flipping the second symbol to non-encoded helps.
                                // Note this is done for both 24 and 25, since the actual window is 25 in RTL.
                                continue;
                                // When we decide to fix the problem and after the flip, the number of symbols exceed  
                                // 24, then in the next 24bit window, we should start with the status of the first 
                                // symbol to flipped. Checked in the next condition.
                            } else if (sumi + sumj > 25) {
                                flipped[0] = fli;
                                num_max_syms = 1;
                                if (flj) {
                                    start_next_window_flipped = 1;
                                } else {
                                    start_next_window_flipped = 0;
                                }
                                goto found_solution_again;
                            }
                            for(int flk=0;flk < 2; flk++) {
                                int sumk = flk ? (1+SIZE_OF_SYMBOL) : origNumofBits[2];
                                if ( (sumi + sumj + sumk  == 24) || (sumi + sumj + sumk  == 25) ) { 
                                    // Continue and check if flipping the third symbol to non-encoded helps.
                                    continue;
                                    // When we decide to fix the problem and after the flip, the number of symbols exceed the 
                                    // 25, then in the next 24bit window, we should start with the status of the first 
                                    // symbol to flipped.
                                } else  if (sumi + sumj + sumk  > 25) {
                                    flipped[0] = fli;
                                    flipped[1] = flj;
                                    num_max_syms = 2;
                                    if (flk) {
                                        start_next_window_flipped = 1;
                                    } else {
                                        start_next_window_flipped = 0;
                                    }
                                    goto found_solution_again;
                                } else {
                                    // < 24
                                    // Sum of first 3 is less than 24, check if the 4th would cross the 
                                    // 24 bit boundary. If no, the max sym should be 4.
                                    flipped[0] = fli;
                                    flipped[1] = flj;
                                    flipped[2] = flk;
                                    if (sumi + sumj + sumk + origNumofBits[3] <= 25) {
                                        num_max_syms = 4;
                                    } else {
                                        num_max_syms = 3;
                                    }
                                    start_next_window_flipped = 0;
                                    goto found_solution_again;
                                }
                            }
                        }
                    }
                    printf("No solution found\n");
                    assert(0);
found_solution_again: ;
                }
                            

                //if(k != inbuf_size[j]-1 || ((j != inbuf_size.size() - 1) && inbuf_size[j+1] != 0))
                //{// this means we should have one extra lookahead byte/symbol available
                    //assert((byte_count + 1) < length);// let's double-check to be sure
                    if (flipped[num_syms_in_window] && 1) 
                    {
                        code.nrOfBits = 0;
                    }
                    num_bits_in_window += code.nrOfBits ? (1+code.nrOfBits) : (1+SIZE_OF_SYMBOL);
                    num_syms_in_window++;
                    if(num_bits_in_window > 25) {
                    //Case where we have to reset the num bits to the last symbol size
                        num_bits_in_window = code.nrOfBits ? (1+code.nrOfBits) : (1+SIZE_OF_SYMBOL);
                    } else if ((num_bits_in_window == 25) || (num_syms_in_window == 4)) {
                    // Reset to 0
                        num_bits_in_window = 0;
                    }
                    //-------------
                    if(num_syms_in_window == num_max_syms)
                    {
                        assert((num_bits_in_window<=25));
                        num_syms_in_window = 0;
                        window_number_in_pipe++;
                    }
                    assert(num_syms_in_window < 4);
                //}
                encodedValues.addInt((code.nrOfBits != 0), 1);              // This is where we add the start bit ('0' if not an ensoded symbol, else '1')
                incrBits += 1;
                 Log(5, "Start bit = %d added, Current bit length: %0d", (code.nrOfBits != 0), incrBits);
                if(code.nrOfBits != 0) {                                    // symbol was encoded
                    int codeReversed = reverseBits_short(code.code) >> (16 - code.nrOfBits);
                    encodedValues.addInt(codeReversed, code.nrOfBits);         // add the encoded symbol
                    incrBits += code.nrOfBits;
                     Log(5, "Byte %4d encoded.\t\tOriginal data: %0x Code: %0x, Reversed: %0x, Code length: %0d, Current bit length: %0d", byte_o, Bytes[byte_o] & mask, code.code, codeReversed, code.nrOfBits, incrBits);
                }
                else {
                    encodedValues.addInt(Bytes[byte_o] & mask, SIZE_OF_SYMBOL);  // add the non encoded symbol
                    incrBits += SIZE_OF_SYMBOL;
                    Log(5, "Byte %4d not encoded.\tOriginal data: %0x, Current bit length: %0d", byte_o, Bytes[byte_o], incrBits);
                }
                if (k == inbuf_size[j]-1) {
                    encodedValues.addInt(0, pipe_padding[j]);                    // add pading to the end of each pipe
                    incrBits += pipe_padding[j];
                    Log(5, "Adding padding of %0d bits for pipe %0d, Current bit length: %0d", pipe_padding[j], j, incrBits);
                }
                byte_o++;
            }
        }
        Log(2, "\nwriteEncodedData: Total size of encoded data to be written = %d \n", encodedValues.length);  // all encoded data (including padding) + Metadata
//        assert(encodedValues.length == sumOfBits);      // TODO update check // check the data size is correct 
    }
    
    if (!statsOnly) {
        int status = 0;
        if ( outputDataRouting == WRITE_TO_FILE )
        {
            if((status = writeToFile(encodedValues.bits.data(), encodedValues.bits.size(), outputDataFileName)));
        }
        else
        {
            if((status = writeToBuffer(encodedValues.bits.data(), encodedValues.bits.size(), outputDataBuffer)));
        }
    }
    return (encodedValues.length);
}

unsigned char Huffman::decodeSymbol(string Huffman, const vector<int> &symbolTable,
        const vector<int> &skipTable)
{
    int code_ptr = 1;
    int symbol_ptr = 0;
    int internal_node = 1;
    int cprefix_H = 0;
    stringstream rptStream;

    Huffman = ' ' + Huffman;
    while(Huffman[code_ptr] == '0' || Huffman[code_ptr] == '1')
    {
        internal_node = (internal_node << 1) - skipTable[code_ptr];
        cprefix_H = (cprefix_H << 1) + (Huffman[code_ptr] - '0');
        if(cprefix_H >= internal_node)
        {
            return symbolTable[symbol_ptr + cprefix_H - internal_node];
        }
        symbol_ptr = symbol_ptr + skipTable[code_ptr];
        code_ptr++;
    }

    rptStream << "decodeSymbol: ERROR - Symbol " << Huffman << " not found in symbol table" << endl;
    
    Error(0, rptStream);
    // Log(0, rptStream.str().c_str());
    return '\0';
}

void Huffman::constructOneSidedTree(vector<int> &leaves, HuffmanTuple_t &node, int level, int value)
{
    if (level >= (int) leaves.size())
        return;

    auto insertLeaf = [this, &leaves, &node, &level, &value]()
    {
        decodingTree.push_back(HuffmanTuple_t());
        leaves[level]--;
        string val;
        for (int i = 0; i < level; i++)
            val = (char) ('0' + ((value >> i) & 1)) + val;
        decodingTable.insert(make_pair(val, '\0'));
        Log(5, "constructOneSidedTree: decoded symbol added to decoding table = %s", val.c_str());
    };

    if (node.rightSon == 0 && leaves[level] > 0)
    {
        value = value << 1 | 1;
        insertLeaf();
        node.rightSon = decodingTree.size() - 1;
        value >>= 1;
    }
    if (node.leftSon == 0 && leaves[level] > 0)
    {
        value <<= 1;
        insertLeaf();
        node.leftSon = decodingTree.size() - 1;
        value >>= 1;
    }

    if(node.rightSon == 0)
    {
        decodingTree.push_back(HuffmanTuple_t());
        node.rightSon = decodingTree.size() - 1;
        constructOneSidedTree(leaves, decodingTree[node.rightSon], level + 1, (value << 1) | 1);
    }
    if(node.leftSon == 0)
    {
        decodingTree.push_back(HuffmanTuple_t());
        node.leftSon = decodingTree.size() - 1;
        constructOneSidedTree(leaves, decodingTree[node.leftSon], level + 1, (value << 1));
    }
}

int Huffman::readEncodedDataFromFile(const string &srcFile, const string &dstFile)
{
    uint8_t  *lvpInputDataBuffer   = NULL;
    uint32_t  lviInputBufferLength = 0;
    uint8_t  *lvpOutputDataBuffer  = NULL;
    uint32_t  lviOutputDataLength  = 0;

    lviOutputDataLength = readEncodedData ( srcFile,
                                            dstFile,
                                            lvpInputDataBuffer,
                                            lviInputBufferLength,
                                            lvpOutputDataBuffer,
                                            READ_FROM_FILE,
                                            WRITE_TO_FILE
                                            );

    return lviOutputDataLength;
}

uint32_t Huffman::readEncodedData (  const string              &srcFile,
                                     const string              &dstFile,
                                     uint8_t                   *inputDataBuffer,
                                     const uint32_t            &inputBufferLength,
                                     uint8_t                   *outputDataBuffer,
                                     huffmanInputDataRouting_t  inputDataRouting,
                                     huffmanOutputDataRouting_t outputDataRouting
                                     )
{
    FILE            *fin;
    FILE            *fout;
    BitData          encodedValues;
    vector<uint8_t> *outputDataVector;
    uint32_t         outputDataLength;
    stringstream     rptStream;
    uint32_t         lviBlockNumber;
    size_t           odvSizePrev;
        
    Log(5, "readEncodedData: start");

    int d_read;

    if ( inputDataRouting == READ_FROM_FILE )
    {
        fin = fopen(srcFile.c_str(), "rb");
        
        if (!fin) {
            throw Exception(
                PrintToString(
                    "Could not open file %s", srcFile.c_str()));
        }
    }
    else
    {
        fin = NULL;
        assert (inputDataBuffer != NULL);
        assert (inputBufferLength>0);
    }
    
    
    if ( outputDataRouting == WRITE_TO_FILE )
    {
        fout = fopen(dstFile.c_str(), "wb");
        
        if (!fout) {
            throw Exception(
                PrintToString(
                    "Could not open file %s", dstFile.c_str()));
        }
    }
    else
    {
        fout             = NULL;
        outputDataVector = new vector<uint8_t>;
        outputDataVector->clear();
    }

    outputDataLength = 0;
    odvSizePrev = 0;
    lviBlockNumber = 0;
    
    uint_least32_t bytes_read = 0;
    
    while ( ( ( inputDataRouting == READ_FROM_FILE )   && ( !feof(fin) ) ) ||
            ( ( inputDataRouting == READ_FROM_BUFFER ) && ( bytes_read < inputBufferLength ) ) )
    {
        uint_least32_t start_word = bytes_read/32;
        vector<char> bytes;
        decodingTable.clear();

        d_read = 0;
        // mode -- check the first byte for details
        //          then get all the METADATA (different sizes for each modes)
        //          - BYPASS    = 2 bytes
        //          - RLE       = 4 bytes
        //          - Huffman   = 54 bytes

        // read the first byte to determine the decompression mode
        if ( inputDataRouting == READ_FROM_FILE )
        {
            bytes_read += fread(&d_read, 1, 1, fin);
        }
        else
        {
            d_read = inputDataBuffer[bytes_read++];
        }
        Log(2, "readEncodedData: Block = %0d Mode = %x ODV size = %0ld (delta %0ld, d_read 0x%0x, bytes_read %0d)", lviBlockNumber, (d_read & 0x03), outputDataVector->size(), ( outputDataVector->size() - odvSizePrev ), (d_read&0xFFFF), bytes_read);
        odvSizePrev = outputDataVector->size();

        //---------------------------------------------------------------------------------------------------------------------
        // BYPASS mode 
        //---------------------------------------------------------------------------------------------------------------------
        if (d_read & 0x01) { 
            rptStream << "readEncodedData: block " << lviBlockNumber << " being decoded in Bypass mode" << endl;
            Report(1, rptStream);
            Log(1, "readEncodedData: Bypass mode");

            int dataSize;

            // Read in data size from. 6 bits in first byte, 6 bits in second byte
            dataSize = (d_read >> 2) & 0x3F;
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
            dataSize |= (d_read & 0x3F) << 6;
            Log(3, "readEncodedData: Data size read = 0x%03x (%0d)", dataSize, dataSize);

            // write data to file
            uint8_t by_data;
            for(int j = 0; j < dataSize + 1; j++) {
                if ( inputDataRouting == READ_FROM_FILE )
                {
                    bytes_read += fread(&by_data, 1, 1, fin);
                }
                else
                {
                    by_data = inputDataBuffer[bytes_read++];
                }
                Log(5, "readEncodedData: Byte %0d: %02x", j, by_data);
                if ( outputDataRouting == WRITE_TO_FILE )
                {
                    fwrite(&by_data, 1, 1, fout);
                }
                else
                {
                    outputDataVector->push_back(by_data);
                }
            }
            // Can only have one block per DMA word. Data will be padded to 256 bits otherwise
            while ( bytes_read/32 == start_word &&
                    ( ( ( inputDataRouting == READ_FROM_FILE )   && ( !feof(fin) ) ) ||
                      ( ( inputDataRouting == READ_FROM_BUFFER ) && ( bytes_read <= inputBufferLength ) ) ) ) {
                if ( inputDataRouting == READ_FROM_FILE )
                {
                    bytes_read += fread(&d_read, 1, 1, fin);
                }
                else
                {
                    d_read = inputDataBuffer[bytes_read++];
                }
            }
                        
            Log(2, "readEncodedData: finished block %0d with bytes_read %0d", lviBlockNumber, bytes_read );
            lviBlockNumber++;
            continue;
        }

        //---------------------------------------------------------------------------------------------------------------------
        // RLE mode
        //---------------------------------------------------------------------------------------------------------------------
        // If the compression is RLE, applied for only 1 symbol
        if(d_read & 0x02 && !RLE_AS_HUFFMAN) {
            rptStream << "readEncodedData: block " << lviBlockNumber << " being decoded in RLE mode" << endl;
            Report(1, rptStream);
            Log(1, "readEncodedData: RLE mode");

            int dataSize;
            char symbol;

            // Read in data size. 6 bits in first byte, 6 bits in second byte
            dataSize = (d_read >> 2) & 0x3F;
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
            dataSize |= (d_read & 0x3F) << 6;
            Log(3, "readEncodedData: Data size read = 0x%03x (%0d)", dataSize, dataSize);
            
            // Read symbol from metadata. Last 2 bits of first byte, first 6 bits from following
            symbol = (d_read >> 6) & 0x03;
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
            symbol |= (d_read & 0x3F) << 2;
            Log(5, "readEncodedData: RLE symbol = 0x%02X ", (int)symbol & 0xFF);

            bytes.insert(bytes.begin(), dataSize+1, symbol);
            
            if ( outputDataRouting == WRITE_TO_FILE )
            {
                fwrite(bytes.data(), bytes.size(), 1, fout);
            }
            else
            {
                for ( uint32_t i = 0; i < bytes.size(); i++ )
                {
                    outputDataVector->push_back(bytes[i]);
                }
            }

            if ( RLE_BLOCKS_PADDED_TO_32_BYTES )
            {
                // Can only have one block per DMA word. Data will be padded to 256 bits otherwise
                while ( ( bytes_read/32 == start_word ) &&
                        ( ( ( inputDataRouting == READ_FROM_FILE )   && ( !feof(fin) ) ) ||
                          ( ( inputDataRouting == READ_FROM_BUFFER ) && ( bytes_read <= inputBufferLength ) ) ) ) {
                    if ( inputDataRouting == READ_FROM_FILE )
                    {
                        bytes_read += fread(&d_read, 1, 1, fin);
                    }
                    else
                    {
                        d_read = inputDataBuffer[bytes_read++];
                    }
                    Log(5, "Bytes read: %0d. Current word: %0d Start word: %0d", bytes_read, bytes_read/32, start_word);
                }
            }
            else
            {
                bytes_read = bytes_read + 1;
                Log(5, "Bytes read: %0d. Current word: %0d Start word: %0d", bytes_read, bytes_read/32, start_word);
            }
            
            Log(2, "readEncodedData: finished block %0d with bytes_read %0d", lviBlockNumber, bytes_read );
            
            lviBlockNumber++;
            continue;
        }

        //---------------------------------------------------------------------------------------------------------------------
        // If it is Huffman
        //---------------------------------------------------------------------------------------------------------------------
        int             dataSize;
        vector<int>     leafTable;
        vector<int>     symbolTable;
        vector<int>     bufferSizes;

        // Read in data size from 6 bits in first byte, 6 bits in second byte
        dataSize = (d_read >> 2) & 0x3F;
        if ( inputDataRouting == READ_FROM_FILE )
        {
            bytes_read += fread(&d_read, 1, 1, fin);
        }
        else
        {
            d_read = inputDataBuffer[bytes_read++];
        }
        dataSize |= (d_read & 0x3F) << 6;

        // Exit if data size is 0. Padded data to interface width
        if (dataSize == 0) {
            Log(1, "readEncodedData: 0 sized block seen. Exiting");
            break;
        }

        rptStream << "readEncodedData: block " << lviBlockNumber << " being decoded in HUFFMAN mode" << endl;
        Report (1, rptStream);

        Log(1, "readEncodedData: HUFFMAN mode\n"); 
        Log(3, "readEncodedData: Data size read = 0x%03x (%0d)", dataSize, dataSize);

        // Read in leaf table. Last 2 bits of first byte, first 6 bits of following
        leafTable.push_back(0);
        for (unsigned short i=1; i<16; ++i) {
            unsigned int leaf = 0;
            leaf |= (d_read >> 6) & 0x03;
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
            leaf |= (d_read & 0x3F) << 2;
            leafTable.push_back(leaf);
            Log(3, "readEncodedData: Leaf table level %0d - Leaves read = 0x%0x", i, leafTable[i]);
        }

        // Read in symbol table. Last 2 bits of current byte, first 6 of following
        for (unsigned short i=0; i<16; ++i) {
            unsigned int symbol = 0;
            symbol |= (d_read >> 6) & 0x03;
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
            symbol |= (d_read & 0x3F) << 2;
            symbolTable.push_back(symbol);
            Log(3, "readEncodedData: Symbol table %0d = 0x%0x", i, symbolTable[i]);
        }

        // Read in buffer pipe sizes.
        short offset = 6;
        for (unsigned short i=0; i<13; ++i) {
            int bufSize = 0;
            int bitsRead = 0;
            int bitsRemaining = 12 - bitsRead;
            while (bitsRemaining) {
                if (bitsRemaining > 8 - offset) {
                    char byteData = (d_read >> offset) & ((1 << (8-offset)) - 1);
                    bufSize |= byteData << bitsRead & ((1 << (bitsRead + 8 - offset)) - 1);
                    bitsRead += 8 - offset;
                    bitsRemaining = 12 - bitsRead;
                    offset = 0;
                    if ( inputDataRouting == READ_FROM_FILE )
                    {
                        bytes_read += fread(&d_read, 1, 1, fin);
                    }
                    else
                    {
                        d_read = inputDataBuffer[bytes_read++];
                    }
                }
                else {
                    char byteData = (d_read >> offset) & ((1 << bitsRemaining) - 1);
                    bufSize |= (byteData << bitsRead) & 0xFFF;
                    bitsRead += bitsRemaining;
                    offset += bitsRemaining;
                    bitsRemaining = 0;
                }
            }
            bufferSizes.push_back(bufSize);
            Log(3, "readEncodedData: Pipe %0d data size = 0x%0x (%0d)", i, bufferSizes[i], bufferSizes[i]);
        }
        // Read one extra byte to clear padding to 16 bits
        if ( inputDataRouting == READ_FROM_FILE )
        {
            bytes_read += fread(&d_read, 1, 1, fin);
        }
        else
        {
            d_read = inputDataBuffer[bytes_read++];
        }

        // Build decoding table from leaves
        decodingTree.clear();
        decodingTree.push_back(HuffmanTuple_t());
        decodingTree.reserve(256);      // Recommended Fix to Original received model by:- Vesa, Ovidiu
        vector<int> tmp_leafTable (leafTable);
        constructOneSidedTree(tmp_leafTable, decodingTree[0], 1, 0);

        int SIZE_OF_SYMBOL = 8;

        for (uint_fast8_t pipe=0; pipe<13; ++pipe) {
            unsigned int bufferSizeB = bufferSizes[pipe]/8 + (bufferSizes[pipe]%8 != 0) + ((bufferSizes[pipe]%16 != 0 && bufferSizes[pipe]%16 <= 8));
            Log(3, "Pipe byte size: %0d", bufferSizeB);

            // Allocate space for data in bit stream and read data into stream
            encodedValues.bits.clear();
            encodedValues.length = bufferSizes[pipe];
            encodedValues.bits.resize(bufferSizeB);
            
            if ( inputDataRouting == READ_FROM_FILE )
            {
                assert(bytes_read += fread(encodedValues.bits.data(), 1, bufferSizeB, fin) == bufferSizeB);
            }
            else
            {
                
                for ( uint32_t i = bytes_read; i < ( bytes_read + bufferSizeB ); i++ )
                {
                    encodedValues.bits[i-bytes_read] = (inputDataBuffer[i]);
                }
                bytes_read += bufferSizeB;
            }

            bool first = true;  // this is used to indicate the first bit of each symbol IE: the start bit
            string s;

            int i = 0;
            for (i=0; i < bufferSizes[pipe]; ) {
                Log(5, "Bit %0d: %0d", (int)i, encodedValues.getBit(i));
                if (first) {
                    if (encodedValues.getBit(i) == 0) { // Start bit == 0; Symbol not encoded
                        ++i;
                        bytes.push_back( (unsigned char) encodedValues.getBitsVal(i, SIZE_OF_SYMBOL) );
                        Log(5, "Unencoded byte added: 0x%0x", (unsigned char) encodedValues.getBitsVal(i, SIZE_OF_SYMBOL) );
                        i += SIZE_OF_SYMBOL;
                        first = true;
                        continue;
                    }
                    first = false;
                    ++i;
                    Log(5, "Bit %0d: %0d", (int)i, encodedValues.getBit(i));
                }
                s += '0' + encodedValues.getBit(i); // Add 0 or 1 as character
                Log(5, "Current symbol : %s", s.c_str());
                ++i;

                auto it = decodingTable.find(s);
                if (it == decodingTable.end()) {    // Current symbol not found in decode table. Continue to the next bit
                    continue;
                }
                Log(4, "Found symbol : %s", s.c_str());

                it->second = decodeSymbol(s, symbolTable, leafTable);
                bytes.push_back(it->second);
                Log(4, "Byte added: 0x%0x", it->second);

                s = "";
                first = true;
            }
            Log(4, "Finished decoding pipe %0d", (int)pipe);
            assert (first == true);
            assert (i == bufferSizes[pipe]);
        }
        if ( outputDataRouting == WRITE_TO_FILE )
        {
            fwrite(bytes.data(), bytes.size(), 1, fout);
        }
        else
        {
            for ( uint32_t i = 0; i < bytes.size(); i++ )
            {
                outputDataVector->push_back(bytes[i]);
            }
        }

        // Can only have one block per DMA word. Data will be padded to 256 bits otherwise
        while (bytes_read/32 == start_word &&
                    ( ( ( inputDataRouting == READ_FROM_FILE )   && ( !feof(fin) ) ) ||
                      ( ( inputDataRouting == READ_FROM_BUFFER ) && ( bytes_read <= inputBufferLength ) ) ) ) {
            if ( inputDataRouting == READ_FROM_FILE )
            {
                bytes_read += fread(&d_read, 1, 1, fin);
            }
            else
            {
                d_read = inputDataBuffer[bytes_read++];
            }
        }

        Log(2, "readEncodedData: finished block %0d with bytes_read %0d", lviBlockNumber, bytes_read );
        lviBlockNumber++;
    }
    
    if ( outputDataRouting == WRITE_TO_BUFFER )
    {
        outputDataLength = outputDataVector->size();
        
        if ( outputDataLength > 0 )
        {
            memcpy(reinterpret_cast<void*>(outputDataBuffer), reinterpret_cast<void*>(outputDataVector->data()), outputDataLength);
            delete outputDataVector;
        }
        else
        {
            rptStream << "readEncodedData: failed (outputDataVector->size(): " << outputDataVector->size() << ")" << endl;
            Error (0, rptStream);
        }   
    }
    else
    {
        outputDataLength = 0;
        fclose(fout);
    }

    return outputDataLength;
}

void Huffman::PrintSymbolTable(void) const
{
    std::ostringstream symbolTable;
    symbolTable << "  | Byte | Code |Length|";
    for (std::map<std::string, HuffmanCoded_t>::const_iterator i=codedSyms.begin(); i!=codedSyms.end(); ++i) {
        uint32_t byte   = i->first[0] & 0xFF;
        uint32_t code   = i->second.code &0xFF;
        uint32_t length = i->second.nrOfBits;
        symbolTable << endl;
        symbolTable << "  |  " << std::hex << std::setfill('0') << std::setw(SIZE_OF_SYMBOL/4 + (SIZE_OF_SYMBOL%4!=0)) << byte;
        symbolTable << "  |  " << std::hex << std::setfill('0') << std::setw(SIZE_OF_SYMBOL/4 + (SIZE_OF_SYMBOL%4!=0)) << code;
        symbolTable << "  |  " << std::setw(2) << length;
        symbolTable << "  |";
    }
    Log(2, "PrintSymbolTable: Huffman symbol table\n%s", symbolTable.str().c_str());
}

void Huffman::PrintLeavesTable(void) const
{
    std::ostringstream leavesTable;
    std::vector<HuffmanTuple_t> treeVector;

    treeVector.push_back(heap.top());
    for (unsigned int i=0; i<treeVector.size(); ++i)
    {
       if (treeVector[i].leftSon != -1)
       {
           treeVector.push_back(nodes[treeVector[i].leftSon]);
           treeVector.push_back(nodes[treeVector[i].rightSon]);
       }
    }

    leavesTable << "  | Level  | Index  |  Left  | Right  |  Leaf  |";
    for (unsigned int i=0; i<treeVector.size(); ++i)
    {
        leavesTable << endl;
        leavesTable << "  |  " << " " << std::setw(2) << std::setfill(' ') << i << " ";
        leavesTable << "  |  " << std::setw(4) << treeVector[i].index;
        if (treeVector[i].leftSon == -1)
        {
            leavesTable << "  |  " << " -- ";
            leavesTable << "  |  " << " -- ";
            leavesTable << "  |  " << " " << std::hex << std::setw(SIZE_OF_SYMBOL/4 + (SIZE_OF_SYMBOL%4!=0)) << std::setfill('0') << treeVector[i].getFullSymbol() << std::dec << std::setfill(' ') << " ";
        }
        else
        {
            leavesTable << "  |  " << std::setw(4) << treeVector[i].leftSon;
            leavesTable << "  |  " << std::setw(4) << treeVector[i].rightSon;
            leavesTable << "  |  " << " -- ";
        }
        leavesTable << "  |";
    }
    Log(3, "PrintLeavesTable: Huffman leaves table\n%s", leavesTable.str().c_str());
}
