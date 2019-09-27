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
	std::string generatedPrototextFileName;
	std::string caseName;
	const std::string savedTestsPath_;
	std::string failedHardwareRunFileName_;
	std::string failedAccuracyFileName_;
	std::string projectRootPath_;
	std::ofstream hardwareResults;
	std::ofstream accuracyResults;
    std::ostringstream hs;
    std::ostringstream as;


	MCMtest(std::string testType);
	void addParam(std::string opCategory, std::string opKey, std::string opValue);
	void generatePrototxt_2dconv();
	void generatePrototxt_diamond_eltwise();
	void generatePrototxt_diamond_concat();
	void createResultsFiles();
	int execute(const char* cmd);
	void saveResult();
	virtual ~MCMtest();
};

#endif /* TESTS_MCMTEST_H_ */
