#include "drogon/HttpAppFramework.h"
#include "spdlog/spdlog.h"

// Include controllers headers to register with server
#include "controllers/Query.h"
#include "server_lib.h"

void init_logger() {}

void run_server() {
    init_logger();
    drogon::app().addListener("localhost", 8080);

    SPDLOG_INFO("Server listening on localhost:8080:");
    drogon::app().run();
}
