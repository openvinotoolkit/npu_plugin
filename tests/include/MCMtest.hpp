#ifndef TESTS_MCMTEST_H_
#define TESTS_MCMTEST_H_

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "include/mcm/compiler/compilation_unit.hpp"


class MCMtest {

public:

	std::string testDescPath;
	mv::json::Object descriptor;
	std::string caseName;

	MCMtest(std::string testType);
	void addParam(std::string opCategory, std::string opKey, std::string opValue);
	void generatePrototxt();
	void saveResult();
	virtual ~MCMtest();
};

#endif /* TESTS_MCMTEST_H_ */
