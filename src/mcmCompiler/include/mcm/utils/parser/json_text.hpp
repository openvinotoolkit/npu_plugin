#ifndef MV_JSON_TEXT_PARSER_HPP_
#define MV_JSON_TEXT_PARSER_HPP_

#include <fstream>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <map>
#include <regex>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/parsing_error.hpp"

namespace mv
{

    class JSONTextParser : public LogSender
    {

        enum class JSONSymbol
        {
            LBrace,
            RBrace,
            LBracket,
            RBracket,
            Colon,
            Comma,
            String,
            Number,
            True,
            False,
            Null,
            EOFSymbol,
            Invalid
        };

        enum class ParserState
        {
            Start,
            ObjectInit,
            MemberKey,
            KeyValueDelim,
            MemeberValue,
            MemberDelim,
            ObjectFinish,
            ArrayInit,
            Element,
            ElementDelim,
            ArrayFinish
        };

        static const std::map<ParserState, std::map<JSONSymbol, ParserState>> pushdownAutomata_;
        static const std::string fileComposeKeyword_;
        static const std::string enableMergeKey_;

        char *buffer_;
        std::string bufferStr_;
        unsigned bufferLength_;
        std::ifstream inputStream_;
        bool composable_;

        unsigned readStream_();
        std::pair<JSONSymbol, std::string> lexer_();

    public:
        /**
         * @brief Construct a new JSONTextParser object
         * 
         * @param composable Enables extension feature which allows to compose multiple files into one JSON object.
         * If true all occurances of string values prefixed with "@FILE:" will be replaced with the content of file under
         * location specified by the string after prefix ("@FILE@path/to/file"). By assumption it has to contain a valid 
         * textual JSON object definition. Note that usage of this flag dissallows to use string @FILE@ anywhere else than
         * for the file composition feature.
         * @param bufferLength Number of buffered characters while loading the file
         */
        JSONTextParser(bool composable = false, unsigned bufferLength = 64);
        JSONTextParser(const JSONTextParser &) = delete;
        ~JSONTextParser();
        const JSONTextParser & operator=(const JSONTextParser &) = delete;
        bool parseFile(const std::string& fileName, json::Value& outputObject);

        std::string getLogID() const override;

    };

}

#endif // MV_PARSER_JSON_TEXT_HPP_