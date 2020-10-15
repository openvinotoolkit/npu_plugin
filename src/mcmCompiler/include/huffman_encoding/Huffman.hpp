///
/// @file
/// @copyright All code copyright Movidius Ltd 2017, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Huffman encoder/decoder header.
///
///

#ifndef HUFFMAN_HPP
#define HUFFMAN_HPP

#include <stdint.h>

#include <map>
#include <queue>
#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <vector>
#include <functional>


typedef enum {
    WRITE_TO_BUFFER,
    WRITE_TO_FILE
    }  huffmanOutputDataRouting_t;

typedef enum {
    READ_FROM_BUFFER,
    READ_FROM_FILE
    }  huffmanInputDataRouting_t;

/**
 * Structure used as input of Huffman encoder.
 */
struct Symbol
{
    std::string symbol;
    int occurrences = 0;

    /**
     * Constructs a symbol from a character and its number of occurrences.
     * @param [in] cSymbol        - A character used for symbol identifier.
     * @param [in] occurrencesNum - Number of occurrences of the symbol.
     */
    explicit Symbol(const char cSymbol, int occurrencesNum) :
        symbol(std::string(1, cSymbol)),
        occurrences(occurrencesNum)
    {}

    /**
     * Constructs a symbol from a string and its number of occurrences.
     * @param [in] sSymbol        - A character used for symbol identifier.
     * @param [in] occurrencesNum - Number of occurrences of the symbol.
     */
    explicit Symbol(const std::string &sSymbol, int occurrencesNum) :
        symbol(sSymbol),
        occurrences(occurrencesNum)
    {}

    std::string getFullSymbol () const
    {
        std::ostringstream name;
        for (unsigned int i=0; i<symbol.size(); ++i)
        {
            name << std::setfill('0') << std::setw(2) << std::hex << ((int) symbol[i] && 0xFF);
        }
        return name.str();
    }
};

/**
 * Huffman structure used during encoding.
 */
struct HuffmanTuple_t
{
    std::string symbol;
    int occurrences = 0, index = 0, leftSon = 0, rightSon = 0;

    explicit HuffmanTuple_t() {};

    /**
     * Constructs a Huffman symbol from a character.
     * @param [in] cSymbol         - A character used for symbol identifier.
     * @param [in] occurrencesNum  - Number of occurrences of the symbol.
     * @param [in] nodeIndex       - Index of the node in the symbols queue.
     * @param [in] leftSonNode     - Index of the node in the symbols queue which
     *                               is its left son in the Huffman tree.
     *                               -1 if the node is a leaf.
     * @param [in] rightSonNode    - Index of the node in the symbols queue which
     *                               is its right son in the Huffman tree.
     *                               -1 if the node is a leaf.
     */
    explicit HuffmanTuple_t(char cSymbol, int occurrencesNum, int nodeIndex, int leftSonNode,
            int rightSonNode)
    {
        *this = HuffmanTuple_t(std::string(1, cSymbol), occurrencesNum, nodeIndex,
                leftSonNode, rightSonNode);
    }

    /**
     * Constructs a Huffman symbol from any string.
     * @param [in] sSymbol         - A string used for symbol identifier.
     * @param [in] occurrencesNum  - Number of occurrences of the symbol.
     * @param [in] NodeIndex       - Index of the node in the symbols queue.
     * @param [in] leftSonNode     - Index of the node in the symbols queue which
     *                               is its left son in the Huffman tree.
     *                               -1 if the node is a leaf.
     * @param [in] rightSonNode    - Index of the node in the symbols queue which
     *                               is its right son in the Huffman tree.
     *                               -1 if the node is a leaf.
     */
    explicit HuffmanTuple_t(const std::string &sSymbol, int occurrencesNum, int nodeIndex,
            int leftSonNode, int rightSonNode) :
        symbol(sSymbol),
        occurrences(occurrencesNum),
        index(nodeIndex),
        leftSon(leftSonNode),
        rightSon(rightSonNode)
    {}

    bool operator <(const HuffmanTuple_t &b) const
    {
        // If number of occurrences is equal.
        if (this->occurrences == b.occurrences)
        {
            // The shortest, then lexicographical, symbol is chosen.
            if (this->symbol.length() == b.symbol.length())
                return (this->symbol < b.symbol);
            return (this->symbol.length() < b.symbol.length());
        }

        return (this->occurrences < b.occurrences);
    }

    bool operator <=(const HuffmanTuple_t &b) const
    {
        return !(b < *this);
    }

    bool operator >(const HuffmanTuple_t &b) const
    {
        return (b < *this);
    }

    bool operator >=(const HuffmanTuple_t &b) const
    {
        return !(*this < b);
    }

    bool operator ==(const HuffmanTuple_t &b) const
    {
        return (symbol == b.symbol && occurrences == b.occurrences
                && index == b.index && leftSon == b.leftSon
                && rightSon == b.rightSon);
    }

    bool operator !=(const HuffmanTuple_t &b) const
    {
        return !(*this == b);
    }

    std::string getFullSymbol () const
    {
        std::ostringstream name;
        for (unsigned int i=0; i<symbol.size(); ++i)
        {
            name << std::setfill('0') << std::setw(2) << std::hex << ((int) symbol[i] & 0xFF);
        }
        return name.str();
    }
};

/**
 * Results structure for Huffman encoding.
 */
struct HuffmanResult_t
{
    int totalSize, compressedSize;
    double entropySize;

    /**
     * Constructor for Huffman results.
     * @param [in] ts - Total size (in bits) of the result.
     * @param [in] cs - Encoded size (in bits) of the results.
     * @param [in] es - Entropy size (in bits) of the results.
     *                  Considered to be achieved on a perfect/complete Huffman tree.
     */
    HuffmanResult_t(int ts, int cs, double es)
    {
        totalSize = ts;
        compressedSize = cs;
        entropySize = es;
    }
};

/**
 * Structure for a Huffman coded symbol.
 * @param [in] code     - 32-bit natural number, which represents
 *                        the Huffman code for a symbol.
 * @param [in] nrOfBits - Number of bits for the symbol. Required for padded 0.
 */
struct HuffmanCoded_t
{
    uint32_t code, nrOfBits;
};

/**
 * Structure with a bit array, used to store Huffman encoding of a byte array.
 */
struct BitData
{
    std::vector<char> bits;
    unsigned int length = 0;

    /**
     * Sets the next bit into bit array with the desired value.
     * @param [in] val - The binary value to insert.
     */
    void setNextBitVal(int val = 0)
    {
        if (length / 8 == bits.size())
            bits.push_back('\0');
        bits[length / 8] |= (val << length % 8);
        length++;
    }

    /**
     * Return a bit value from a given index in bit array.
     * @param [in] pos - The position of the desired bit.
     * return          - Integer value of the bit.
     */
    int getBit(int pos)
    {
        return (((1LL * bits[pos / 8]) >> (pos % 8)) & 1);
    }

    /**
     * Computes the integer value given by bits [pos:pos+cnt] in bit array.
     * @param [in] pos - The MSB index in bit array of the desired value (little endian).
     * @param [in] cnt - Number of bits of the number.
     * return          - Integer value, that represent the value.
     */
    int getBitsVal(int pos, int cnt)
    {
        int val = 0;

        for (int i = pos; i < pos + cnt; i++)
        {
            val |= getBit(i) << (i-pos);
        }

        return val;
    }

    /**
     * Inserts an integer into a bit array,
     * preserving bit order from left to right (little endian).
     * @param [in] nr     - The number to insert into bits array.
     * @param [in] bitsNr - Number of bits to insert. Required for padding 0.
     */
    void addInt(unsigned int nr, int bitsNr)
    {
        for (int i = 0; i < bitsNr; ++i)
            setNextBitVal((nr >> i) & 1);
    }
};

/**
 * Huffman encoder/decoder class
 */
class Huffman
{
private:
    std::vector<HuffmanTuple_t> nodes;
    std::priority_queue<HuffmanTuple_t, std::vector<HuffmanTuple_t>,
            std::greater<HuffmanTuple_t>> heap;
    std::map<std::string, HuffmanCoded_t> codedSyms;
    std::vector<int> levelLeavesCnt, encSyms, interiorNodes, symAddr, inbuf_size, buff_bit_count, pipe_padding;
    std::vector<std::pair<int, std::string>> encSymLengths;
    std::vector<HuffmanTuple_t> decodingTree;
    std::map<std::string, unsigned char> decodingTable;

    long long sumOfBits, originalSize;
    double sumOfBitsOptimal;
    int maxLevel, nrOfDistSyms;
    unsigned int headerBits, dataSize, mode;
    unsigned int SIZE_OF_SYMBOL;
    bool RLE_BLOCKS_PADDED_TO_32_BYTES;
    bool RLE_AS_HUFFMAN;
    int encodeSymbols;

    static const uint_fast16_t MAX_BUF_BIT_SIZE = 2560;
    /**
     * Recursive Huffman tree Depth First Search traversal.
     * Updates the codes for each symbol.
     * Updates statistics about the compression.
     * @param [in] node  - The current node in Huffman tree, which has 0 or 2 sons.
     * @param [in] steps - The current depth of the node (1 based).
     *                     Also meaning the number of bits used to encode path until now.
     * @param [in] code  - The coding value until the current symbol.
     */
    void DFS(const HuffmanTuple_t &node, int steps, uint32_t code);

    /**
    * Get an encoded symbol from a dictionary.
    * @param [in] sym - The symbol as a string, key in dictionary.
    * return          - The Huffman encoded symbol for the string
    *                   or an empty symbol, if not present.
    */
    HuffmanCoded_t getSymbolCode(const std::string &sym);

    /**
    * Get an encoded symbol from a dictionary.
    * @param [in] sym - The symbol as a character, key in dictionary.
    * return          - The Huffman encoded symbol for the string
    *                   or an empty symbol, if not present.
    */
    HuffmanCoded_t getSymbolCode(char sym);

    /**
     * Build a min-heap from vector of symbol frequencies.
     * @param [in] data - A vector of symbols, described by
     *                      * string representation
                            * number of occurrences.
     * @param [in] bpb  - Number of bits per symbol.
     */
    void constructHeap(const std::vector<Symbol> &data, int bpb);

    void generateEncodedSymbols();

    void constructOneSidedTree(std::vector<int> &leaves, HuffmanTuple_t &node,
            int level, int value);

    unsigned char decodeSymbol(std::string Huffman, const std::vector<int> &symbolTable,
            const std::vector<int> &skipTable);

    void insertLeaf(std::vector<int> &leaves, HuffmanTuple_t &node, int level, int value);

public:
    /**
     * Constructor runs reset() method; no dynamic allocation so no memory to allocate
     */
    Huffman();

    /**
     * Constructor with initializer of maximum symbols allowed to be encoded.
     * @param [in] nrOfEncodedSymbol - Maximum number of encoded symbols.
     */
    Huffman(int nrOfEncodedSymbol);
    
    /**
     * Constructor runs reset() method; no dynamic allocation so no memory to allocate
     */
    ~Huffman();

    /**
     * The encoding function, which computes the Huffman tree, table, and statistics.
     * @param [in] data - Array of symbols, described by
     *                      * string symbol
     *                      * number of occurrences in original data.
     * @param [in] len  - Length of the array.
     * @param [in] bpb  - Bits per symbol in the original data.
     * @param [in] bypass   - enable bypass mode.
     * return           - Returns a structure with Huffman encoding statistics.
     */
    HuffmanResult_t encode(const Symbol *data, int len, int bpb, bool bypass);

    /**
     * Computes the Huffman tree, table, and statistics.
     * @param [in] data - Vector of symbols, described by
     *                      * string symbol
     *                      * number of occurrences in original data.
     * @param [in] bpb  - Bits per symbol in the original data.
     * @param [in] bypass   - enable bypass mode.
     * return           - Returns a structure with Huffman encoding statistics.
     */
    HuffmanResult_t encode(const std::vector<Symbol> &data, int bpb, bool bypass);

    /**
     * Writes the header and encoded data to a file, based on a previous computed table.
     * @param [in] data     - Array of bytes. Each byte is looked in a previous computed table.
     * @param [in] length   - Number of bytes to write from the array.
     * @param [in] filename - Name of the output file.
     * @param [in] bypass   - enable bypass mode.
     * return               - 0 on success, negative value otherwise.
     */
    int writeEncodedDataToFile(const void *data, int length,
            const std::string &filename, bool bypass, bool statsOnly);

    /**
     * Writes the header and encoded data to a file, based on a previous computed table.
     * @param [in] data     - Array of bytes. Each byte is looked in a previous computed table.
     * @param [in] length   - Number of bytes to write from the array.
     * @param [in] outputDataFileName - Name of the output file.
     * @param [in] outputDataBuffer - Name of the output byte buffer.
     * @param [in] bypass   - enable bypass mode.
     * @param [in] outputDataRouting - where to route output data (file or buffer)
     * return               - 0 on success, negative value otherwise.
     */
     int writeEncodedData( const void *data,
                           int length,
                           const std::string &outputDataFileName,
                           std::vector<char> *outputDataBuffer,
                           bool bypass,
                           bool statsOnly,
                           huffmanOutputDataRouting_t &outputDataRouting
                           );


    /**
     * Clears all Huffman data.
     */
    void reset();

    /**
     * Creates a vector of distinct symbols, with a dictionary structure like:
     *  * key   - string representation of a symbol
     *  * value - number of occurrences
     * @param [in] data     - Array of bytes.
     * @param [in] length   - Number of bytes to process from the array.
     * return               - Vector with mapped symbols.
     */

    static std::vector<Symbol> getSymFreqs(const void *data, int length, int bits = 8);

    /**
     * Reads an encoded file and writes to another file all the decoded contents.
     * @param [in] srcFile - Encoded file name.
     * @param [in] dstFile - Destination of decoded file name.
     * return              - 0 on success, negative value otherwise.
     */
    int readEncodedDataFromFile(const std::string &srcFile,
                                const std::string &dstFile);

    /**
     * Core of readEncodedDataFromFile
     * @param [in] srcFile - Encoded file name.
     * @param [in] dstFile - Destination of decoded file name.
     * @param [in] inputDataBuffer - input data buffer
     * @param [in] inputBufferLength - size of input data buffer (bytes)
     * @param [in] outputDataLength - size of output data buffer (bytes)
     * @param [in] inputDataRouting - routing of input data (buffer/file)
     * @param [in] outputDataRouting - routing of output data (buffer/file)
     * 
     * return              - NULL if writing to file, else address of outputData buffer (uint8_t)
     */
    uint32_t readEncodedData (  const std::string         &srcFile,
                                const std::string         &dstFile,
                                uint8_t                   *inputDataBuffer,
                                const uint32_t            &inputBufferLength,
                                uint8_t                   *outputDataBuffer,
                                huffmanInputDataRouting_t  inputDataRouting,
                                huffmanOutputDataRouting_t outputDataRouting
                                );

    void PrintSymbolTable (void) const;
    void PrintLeavesTable (void) const;
};

#endif //HUFFMAN_HPP
