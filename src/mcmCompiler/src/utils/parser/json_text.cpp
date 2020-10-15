#include "include/mcm/utils/parser/json_text.hpp"

#include <cctype>
#include <iostream>

const std::map<mv::JSONTextParser::ParserState, std::map<mv::JSONTextParser::JSONSymbol, mv::JSONTextParser::ParserState>>
    mv::JSONTextParser::pushdownAutomata_ =
{
    {
        ParserState::Start,
        {
            { JSONSymbol::LBrace, ParserState::ObjectInit },
            { JSONSymbol::LBracket, ParserState::ArrayInit }
        }
    },
    {
        ParserState::ObjectInit,
        {
            { JSONSymbol::RBrace, ParserState::ObjectFinish },
            { JSONSymbol::String, ParserState::MemberKey}
        }
    },
    {
        ParserState::MemberKey,
        {
            { JSONSymbol::Colon, ParserState::KeyValueDelim }
        }
    },
    {
        ParserState::KeyValueDelim,
        {
            { JSONSymbol::String, ParserState::MemeberValue },
            { JSONSymbol::False, ParserState::MemeberValue },
            { JSONSymbol::True, ParserState::MemeberValue },
            { JSONSymbol::Null, ParserState::MemeberValue },
            { JSONSymbol::Number, ParserState::MemeberValue },
            { JSONSymbol::LBrace, ParserState::ObjectInit },
            { JSONSymbol::LBracket, ParserState::ArrayInit }
        }
    },
    {
        ParserState::MemeberValue,
        {
            { JSONSymbol::Comma, ParserState::MemberDelim },
            { JSONSymbol::RBrace, ParserState::ObjectFinish }
        }
    },
    {
        ParserState::MemberDelim,
        {
            { JSONSymbol::String, ParserState::MemberKey }
        }
    },
    {
        ParserState::ArrayInit,
        {
            { JSONSymbol::String, ParserState::Element },
            { JSONSymbol::False, ParserState::Element },
            { JSONSymbol::True, ParserState::Element },
            { JSONSymbol::Null, ParserState::Element },
            { JSONSymbol::Number, ParserState::Element },
            { JSONSymbol::LBrace, ParserState::ObjectInit },
            { JSONSymbol::LBracket, ParserState::ArrayInit },
            { JSONSymbol::RBracket, ParserState::ArrayFinish }
        }
    },
    {
        ParserState::Element,
        {
            { JSONSymbol::Comma, ParserState::ElementDelim },
            { JSONSymbol::RBracket, ParserState::ArrayFinish }
        }
    },
    {
        ParserState::ElementDelim,
        {
            { JSONSymbol::String, ParserState::Element },
            { JSONSymbol::False, ParserState::Element },
            { JSONSymbol::True, ParserState::Element },
            { JSONSymbol::Null, ParserState::Element },
            { JSONSymbol::Number, ParserState::Element },
            { JSONSymbol::LBrace, ParserState::ObjectInit },
            { JSONSymbol::LBracket, ParserState::ArrayInit }
        }
    }
};

mv::JSONTextParser::JSONTextParser(unsigned bufferLength) :
bufferLength_(bufferLength)
{
    if (bufferLength_ == 0)
        throw ArgumentError(*this, "bufferLength", std::to_string(bufferLength_), "Defined as 0");

    buffer_ = new char[bufferLength_];
}

mv::JSONTextParser::~JSONTextParser()
{
    delete [] buffer_;
}

unsigned mv::JSONTextParser::readStream_()
{
    inputStream_.read(buffer_, bufferLength_);
    int charsCount = inputStream_.gcount();
    std::string newContent(buffer_, charsCount);
    newContent.erase(
        std::remove_if(
            newContent.begin(),
            newContent.end(),
            [](const char c) { return std::isspace(static_cast<unsigned char>(c)); }),
        newContent.end());
    bufferStr_ += newContent;
    return charsCount;
}

std::pair<mv::JSONTextParser::JSONSymbol, std::string> mv::JSONTextParser::lexer_()
{

    while (inputStream_ && bufferStr_.empty())
        readStream_();

    if (bufferStr_.empty())
        return {JSONSymbol::EOFSymbol, ""};

    std::string content;

    switch(bufferStr_[0])
    {
        case '{':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::LBrace, content};

        case '}':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::RBrace, content};

        case '[':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::LBracket, content};

        case ']':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::RBracket, content};

        case ':':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::Colon, content};

        case ',':
            content = bufferStr_[0];
            bufferStr_.erase(0,1);
            return {JSONSymbol::Comma, content};

        case '\"':
        {

            bufferStr_.erase(0,1);

            std::size_t found = bufferStr_.find('\"');
            while (found == std::string::npos)
            {
                if (inputStream_)
                {
                    readStream_();
                    found = bufferStr_.find('\"');
                }
                else
                    return {JSONSymbol::EOFSymbol, ""};
            }

            content += bufferStr_.substr(0, found);
            bufferStr_.erase(0, found + 1);

            found = content.find_first_of("\\\b\f\n\r\t");
            if (found != std::string::npos)
                return {JSONSymbol::Invalid, content};

            return {JSONSymbol::String, content};

        }

        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
        case '-':
        {

            std::regex numEx("-?(?:0|[1-9][[:digit:]]*)(?:\\.[[:digit:]]+)?(?:[eE][+-]?[[:digit:]]+)?");
            std::regex Ex("[,}\\]]");
            std::smatch match;

            if (!std::regex_search(bufferStr_, match, numEx))
                return {JSONSymbol::Invalid, bufferStr_};

            while (bufferStr_.length() <= (unsigned)(match.position(0) + match.length(0)) ||
                !std::regex_match(bufferStr_.substr((unsigned)(match.position(0) + match.length(0)), 1), Ex))
            {
                if (inputStream_)
                {
                    readStream_();
                    if (!std::regex_search(bufferStr_, match, numEx))
                        return {JSONSymbol::Invalid, bufferStr_};
                }
                else
                {
                    return {JSONSymbol::EOFSymbol, ""};
                }
            }

            content += bufferStr_.substr(0, match.position(0) + match.length(0));
            bufferStr_.erase(0, match.position(0) + match.length(0));
            return {JSONSymbol::Number, content};

        }

        case 'n':
        case 't':
        case 'f':
        {

            std::string keywordStr;
            JSONSymbol keywordSymbol;

            if (bufferStr_[0] == 'n')
            {
                keywordStr = "null";
                keywordSymbol = JSONSymbol::Null;
            }
            else if (bufferStr_[0] == 't')
            {
                keywordStr = "true";
                keywordSymbol = JSONSymbol::True;
            }
            else
            {
                keywordStr = "false";
                keywordSymbol = JSONSymbol::False;
            }
            while (bufferStr_.length() < keywordStr.length() + 1 && inputStream_)
                readStream_();

            if (!inputStream_ && bufferStr_.length() < keywordStr.length() + 1)
                return {JSONSymbol::EOFSymbol, ""};

            if (bufferStr_.compare(0, keywordStr.length() + 1, keywordStr) &&
                (bufferStr_[keywordStr.length()] == '}' || bufferStr_[keywordStr.length()] == ']' ||
                bufferStr_[keywordStr.length()] == ','))
            {

                content = bufferStr_.substr(0, keywordStr.length());
                bufferStr_.erase(0, content.length());
                return {keywordSymbol, content};

            }

            return {JSONSymbol::Invalid, ""};

        }

    }

    return {JSONSymbol::Invalid, ""};

}


bool mv::JSONTextParser::parseFile(const std::string& fileName, json::Value& outputObject)
{

    inputStream_.open(fileName);

    if (!inputStream_)
        return false;

    bufferStr_.clear();

    ParserState currentState = ParserState::Start;
    std::stack<std::pair<JSONSymbol, std::string>> symbolStack;
    json::Value root;
    std::stack<json::Value*> jsonHierarchy;
    std::string lastKey;

    for(auto currentSymbol = lexer_(); currentSymbol.first != JSONSymbol::EOFSymbol; currentSymbol = lexer_())
    {

        if (currentSymbol.first == JSONSymbol::Invalid)
            throw ParsingError(*this, fileName, "Invalid symbol \"" + currentSymbol.second + "\"");

        if (pushdownAutomata_.find(currentState) == pushdownAutomata_.end())
        {
            if (currentState == ParserState::ArrayFinish || currentState == ParserState::ObjectFinish)
            {

                if (symbolStack.top().first == JSONSymbol::LBrace)
                    currentState = ParserState::MemeberValue;
                else if (symbolStack.top().first == JSONSymbol::LBracket)
                    currentState = ParserState::Element;
                else
                    throw ParsingError(*this, fileName, "Syntax error");

            }
            else
                throw ParsingError(*this, fileName, "Syntax error");
        }

        if (pushdownAutomata_.at(currentState).find(currentSymbol.first) == pushdownAutomata_.at(currentState).end())
                throw ParsingError(*this, fileName, "Syntax error");

        currentState = pushdownAutomata_.at(currentState).at(currentSymbol.first);

        if (currentState == ParserState::MemberKey)
            lastKey = currentSymbol.second;
        else if (currentState == ParserState::MemeberValue || currentState == ParserState::Element)
        {
            json::Value newValue;
            switch (currentSymbol.first)
            {

                case JSONSymbol::Number:
                    try
                    {
                        std::size_t intPos, doublePos;
                        long long intVal = std::stoll(currentSymbol.second, &intPos);
                        double doubleVal = std::stof(currentSymbol.second, &doublePos);

                        if (doublePos > intPos)
                            newValue = doubleVal;
                        else
                            newValue = intVal;
                    }
                    catch (std::invalid_argument& e)
                    {
                       throw ParsingError(*this, fileName, "Invalid numeric value \"" + currentSymbol.second + "\"");
                    }
                    break;


                case JSONSymbol::String:
                    newValue = currentSymbol.second;
                    break;

                case JSONSymbol::True:
                    newValue = true;
                    break;

                case JSONSymbol::False:
                    newValue = false;
                    break;

                case JSONSymbol::Null:
                    break;

                default:
                    throw ParsingError(*this, fileName, "Invalid value \"" + currentSymbol.second + "\"");

            }

            if (jsonHierarchy.top()->valueType() == json::JSONType::Array)
                jsonHierarchy.top()->append(newValue);
            else
                jsonHierarchy.top()->emplace({lastKey, newValue});

        }

        if (currentSymbol.first == JSONSymbol::LBrace)
        {
            if (symbolStack.empty())
            {
                root = json::Object();
                jsonHierarchy.push(&root);
            }
            else
            {
                if (jsonHierarchy.top()->valueType() == json::JSONType::Array)
                {
                    jsonHierarchy.top()->append(json::Object());
                    jsonHierarchy.push(&(*jsonHierarchy.top())[jsonHierarchy.top()->size() - 1]);
                }
                else
                {
                   jsonHierarchy.top()->emplace({lastKey, json::Object()});
                   jsonHierarchy.push(&(*jsonHierarchy.top())[lastKey]);
                }
            }

            symbolStack.push(currentSymbol);

        }
        else if (currentSymbol.first == JSONSymbol::LBracket)
        {
            if (symbolStack.empty())
                root = json::Array();
            else
            {
                if (jsonHierarchy.top()->valueType() == json::JSONType::Array)
                {
                    jsonHierarchy.top()->append(json::Array());
                    jsonHierarchy.push(&(*jsonHierarchy.top())[jsonHierarchy.top()->size() - 1]);
                }
                else
                {
                   jsonHierarchy.top()->emplace({lastKey, json::Array()});
                   jsonHierarchy.push(&(*jsonHierarchy.top())[lastKey]);
                }
            }

            symbolStack.push(currentSymbol);
        }
        else if (currentSymbol.first == JSONSymbol::RBrace)
        {
            if (symbolStack.top().first == JSONSymbol::LBrace)
            {
                symbolStack.pop();
                jsonHierarchy.pop();
            }
            else
                throw ParsingError(*this, fileName, "Incorrect right brace");
        }
        else if (currentSymbol.first == JSONSymbol::RBracket)
        {
            if (symbolStack.top().first == JSONSymbol::LBracket)
            {
                symbolStack.pop();
                jsonHierarchy.pop();
            }
            else
                throw ParsingError(*this, fileName, "Incorrect right bracket");
        }
    }

    if (!symbolStack.empty())
    {
        switch (symbolStack.top().first)
        {
            case JSONSymbol::LBrace:
                throw ParsingError(*this, fileName, "Unterminated left brace");
            case JSONSymbol::LBracket:
                throw ParsingError(*this, fileName, "Unterminated left bracket");
            default:
                throw ParsingError(*this, fileName, "Unexpected symbol \"" + symbolStack.top().second + "\"");
        }
    }

    outputObject = root;
    return true;

}

std::string mv::JSONTextParser::getLogID() const
{
    return "JSONTextParser";
}
