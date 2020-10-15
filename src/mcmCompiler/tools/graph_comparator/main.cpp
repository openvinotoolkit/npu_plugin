#include <iostream>
#include "include/graph_comparator/graph_comparator.hpp"

int main(int argc, char* argv[])
{

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " graph_file1 graph_file2 [-v]" << std::endl;
        return 2;
    }

    if (argc == 4)
    {
        if (std::string(argv[3]) != "-v")
        {
            std::cerr << "Usage: " << argv[0] << " graph_file1 graph_file2 [-v]" << std::endl;
            return 2;
        }

        mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    }
    else
        mv::Logger::setVerboseLevel(mv::VerboseLevel::Silent);

    mv::tools::GraphComparator gc;

    mv::Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator",
        "Comparing - " + std::string(argv[1]) + " and " + std::string(argv[2]));

    bool result;
    try
    {
        result = gc.compare(argv[1], argv[2]);
    }
    catch (const mv::ArgumentError&)
    {
        return 2;
    }

    if (result)
        mv::Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator",
            "Identical - " + std::string(argv[1]) + " and " + std::string(argv[2]));
    else
    {

        mv::Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator",
            "Different - " + std::string(argv[1]) + " and " + std::string(argv[2]));

        for (auto it = gc.lastDiff().begin(); it != gc.lastDiff().end(); ++it)
            mv::Logger::log(mv::Logger::MessageType::Info, "tools:GraphComparator", "Diff: " + *it);

        return 1;

    }

    return 0;

}