#include "hetero_vpux_sqlite.hpp"
#include <assert.h>

SqliteSupport::SqliteSupport(std::string _dbname, std::string _network): dbname(_dbname), network(_network) {
    int sres = sqlite3_open(dbname.c_str(), &DB);

    if (sres) {
        std::ostringstream err;
        err << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
        throw std::runtime_error(err.str());
    } else {
        std::cout << "Opened Database Successfully! " << dbname << std::endl;
    }
    createTableIfNotExists();
}

SqliteSupport::~SqliteSupport() {
    int sres = sqlite3_close(DB);
    if (sres) {
        std::ostringstream err;
        err << "Error close DB " << sqlite3_errmsg(DB) << std::endl;
        std::cerr << err.str() << std::endl;
    } else {
        std::cout << "Closed Database Successfully!" << std::endl;
    }
}

void SqliteSupport::flush() {
    int sres = sqlite3_db_cacheflush(DB);
    if (sres != SQLITE_OK) {
        std::ostringstream err;
        err << "Error flush DB: " << sres << std::endl;
        throw std::runtime_error(err.str());
    } else {
        std::cout << "Flushed Database Successfully!" << std::endl;
    }
}

void SqliteSupport::createTableIfNotExists() {
    std::string sql = "CREATE TABLE IF NOT EXISTS " + network +
                      " ("
                      "ID INTEGER PRIMARY KEY AUTOINCREMENT    NOT NULL, "
                      "NUM       INT     NOT NULL, "
                      "LAYER          TEXT    NOT NULL, "
                      "START_TIME       INT     NOT NULL, "
                      "FINISH_TIME       INT, "
                      "STATUS           INT NOT NULL);";

    char* messaggeError;
    std::cout << "SQL exec: " << sql << std::endl;

    int sres = sqlite3_exec(DB, sql.c_str(), NULL, 0, &messaggeError);

    if (sres != SQLITE_OK) {
        std::ostringstream err;
        err << "Error Create Table: " << messaggeError << "; errcode " << sqlite3_errcode(DB) << "/"
            << sqlite3_extended_errcode(DB) << std::endl;
        sqlite3_free(messaggeError);
        throw std::runtime_error(err.str());
    } else {
        std::cout << "Table created Successfully" << std::endl;
    }
}

int g_lastLayerRecords = 0;
std::string g_layer;
int64_t g_startTime, g_finishTime;
inferStateEnum g_inferState;

static int callback(void* data, int argc, char** argv, char** azColName) {
    int i;
    fprintf(stderr, "%s: ", (const char*)data);

    for (i = 0; i < argc; i++) {
        printf("[%d] %s = %s\n", i, azColName[i], argv[i] ? argv[i] : "NULL");
    }

    printf("\n");

    assert(argc == 6);

    int c = 1;
    g_layer = argv[++c];
    g_startTime = atoi(argv[++c]);
    g_finishTime = argv[++c] ? atoi(argv[c]) : 0;
    g_inferState = inferStateEnum(atoi(argv[++c]));

    ++g_lastLayerRecords;

    return 0;
}

bool SqliteSupport::getLastLayerStarted(std::string& layer, int64_t& startTime, int64_t& finishTime,
                                        inferStateEnum& inferState) {
    std::string query =
            "SELECT * FROM " + network + " WHERE START_TIME=(SELECT MAX(START_TIME) FROM " + network + "); ";
    g_lastLayerRecords = 0;
    sqlite3_exec(DB, query.c_str(), callback, NULL, NULL);
    if (g_lastLayerRecords == 1) {
        layer = g_layer;
        startTime = g_startTime;
        finishTime = g_finishTime;
        inferState = g_inferState;
        return true;
    } else if (g_lastLayerRecords == 0) {
        return false;
    }

    throw std::runtime_error("More than one last record matched");
}

void SqliteSupport::insertLayer(int number, const std::string& layer, int64_t startTime) {
    std::ostringstream sql_ins;
    sql_ins << "INSERT INTO " << network << " (NUM, LAYER, START_TIME, FINISH_TIME, STATUS) "
            << " VALUES ( " << number << ", '" << layer << "', " << startTime << ", NULL, "
            << (int)inferStateEnum::TO_BE_RUN << ");";

    char* messageError;
    std::cout << "SQL exec: " << sql_ins.str() << std::endl;
    int sres = sqlite3_exec(DB, sql_ins.str().c_str(), NULL, 0, &messageError);
    if (sres != SQLITE_OK) {
        std::ostringstream err;
        err << "Error Insert: " << messageError << "; errcode " << sqlite3_errcode(DB) << "/"
            << sqlite3_extended_errcode(DB) << std::endl;
        sqlite3_free(messageError);
        throw std::runtime_error(err.str());
    } else
        std::cout << "Records created Successfully!" << std::endl;
}

void SqliteSupport::updateLayer(int64_t startTime, int64_t finishTime, inferStateEnum inferState) {
    std::ostringstream sql_upd;
    sql_upd << "UPDATE " << network << " SET FINISH_TIME=" << finishTime << ", STATUS=" << (int)inferState
            << " WHERE START_TIME=" << startTime << ";";

    char* messageError;
    std::cout << "SQL exec: " << sql_upd.str() << std::endl;
    int sres = sqlite3_exec(DB, sql_upd.str().c_str(), NULL, 0, &messageError);
    if (sres != SQLITE_OK) {
        std::ostringstream err;
        err << "Error update: " << messageError << "; errcode " << sqlite3_errcode(DB) << "/"
            << sqlite3_extended_errcode(DB) << std::endl;
        sqlite3_free(messageError);
        throw std::runtime_error(err.str());
    } else
        std::cout << "Records updated  Successfully!" << std::endl;
}
