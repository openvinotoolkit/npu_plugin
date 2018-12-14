
#include <gtest/gtest.h>

//global file names 
namespace {
std::string file_name_one;
std::string file_name_two;
}

class TestEnvironment : public testing::Environment {
    
 public:

  explicit TestEnvironment(const std::string &command_line_arg_one, const std::string &command_line_arg_two) {
      
    file_name_one = command_line_arg_one;
    file_name_two = command_line_arg_two;
  }
};
