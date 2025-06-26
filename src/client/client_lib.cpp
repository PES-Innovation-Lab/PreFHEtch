#include "spdlog/spdlog.h"
#include "cpr/cpr.h"

#include "client_lib.h"

void ping_server() {
    SPDLOG_INFO("Pinging server on localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url("http://localhost:8080/ping"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);
}
