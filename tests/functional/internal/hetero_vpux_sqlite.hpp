#pragma once

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>

enum class inferStateEnum { TO_BE_RUN = 1, INFERRED_OK, FAIL_EXCEPTION, FAIL_SIGSEG, FAIL_REBOOT };

class SqliteSupport {
public:
    SqliteSupport(std::string _dbname, std::string _network);
    bool getLastLayerStarted(std::string& layer, int64_t& startTime, int64_t& finishTime, inferStateEnum& inferState);
    void updateLayer(int64_t startTime, int64_t finishTime, inferStateEnum inferState);
    void insertLayer(int number, const std::string& layer, int64_t startTime);
    void flush();
    ~SqliteSupport();

private:
    sqlite3* DB;
    const std::string network;
    std::string dbname;

    void createTableIfNotExists();
};
