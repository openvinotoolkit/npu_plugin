#pragma once

#include "gtest/gtest.h"
#include "mcm/compiler/compilation_descriptor.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

template<typename T>
class LayersTest: public testing::TestWithParam<T>
{
private:
    std::string testname;

protected:
    std::string testGetName() const
    {
        return testname;
    }

    void testSetName(const std::string& name)
    {
        testname = name;
    }

    std::string testToString(const mv::Target& target)
    {
        assert(target == mv::Target::ma2480 || target == mv::Target::ma2490);
        std::string targetName = target == mv::Target::ma2480 ? "ma2480" : "ma2490";
        return targetName;
    }

    std::string testToString(const mv::Shape& shape)
    {
        std::string str = std::to_string(shape[0]);
        for (unsigned i=1; i < shape.ndims(); i++)
        {
            str += "x" + std::to_string(shape[i]);
        }
        return str;
    }

    template<typename S>
    std::string testToString(const std::vector<S>& mask)
    {
        assert(mask.size() > 0);
        std::string str = std::to_string(mask[0]);
        for (unsigned i=1; i < mask.size(); i++)
        {
            str += "x" + std::to_string(mask[i]);
        }
        return str;
    }

    mv::CompilationDescriptor& testGetCompilationDescriptor(mv::CompilationUnit& unit,
                                                            const mv::Target& target)
    {
        std::string targetName = testToString(target);
        std::string compDescPath = mv::utils::projectRootPath()
                                 + "/config/compilation/debug_" + targetName + ".json";
        unit.loadCompilationDescriptor(compDescPath);
        auto &compDesc = unit.compilationDescriptor();
        return compDesc;
    }

    std::string testSetGenBlob(mv::CompilationDescriptor &compDesc)
    {
        try
        {
            //-----------------------------------------------------------------------
            // Note: MCM compiler does not support generating blob for Keem Bay (yet)
            //-----------------------------------------------------------------------
            compDesc.setPassArg("GenerateBlob", "fileName", testGetName() + ".blob");
            compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
            compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

            return "OK";
        }
        catch (...)
        {
            return "ERROR: cannot set arguments for blob generation!";
        }
    }

    std::string testSetGenDot(mv::CompilationDescriptor &compDesc)
    {
        try
        {
            //--------------------------------------------------------------------------
            // Setting scope to control-model disables compute model details in dot-file
            //--------------------------------------------------------------------------
            // compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));
            compDesc.setPassArg("GenerateDot", "content", std::string("full"));
            compDesc.setPassArg("GenerateDot", "html", true);

            //-------------------------------------------------------------------------
            // MCM compiler dumps only final model, if we setup output file's name here
            //-------------------------------------------------------------------------
            // compDesc.setPassArg("GenerateDot", "output", "layers_" + testGetName() + ".dot");

            return "OK";
        }
        catch (...)
        {
            return "ERROR: cannot set arguments for *.dot generation!";
        }
    }

    std::string testDumpJson(const mv::Element& result)
    {
        std::fstream file_out("layers_" + testGetName() + ".json", std::fstream::out);
        file_out << result.toJSON().stringifyPretty() << std::endl;
        file_out.close();
        return "OK";
    }

    // TODO: implement dumping of blob file (e.g. renaming if necessary)
    std::string testDumpBlob()
    {
        return "OK";
    }

    std::string testDumpDot()
    {
        auto name = testGetName();

        rename("original_model.dot", (name + "_original.dot").c_str());
        rename("adapt_model.dot"   , (name +    "_adapt.dot").c_str());
        rename("final_model.dot"   , (name +          ".dot").c_str());

        // TODO: rework this UNIX-specific code to enable Windows
        system(("dot -Tpng " + name + "_original.dot -o " + name + "_original.png").c_str());
        system(("dot -Tpng " + name +    "_adapt.dot -o " + name +    "_adapt.png").c_str());
        system(("dot -Tpng " + name +          ".dot -o " + name +          ".png").c_str());

        return "OK";
    }
};
