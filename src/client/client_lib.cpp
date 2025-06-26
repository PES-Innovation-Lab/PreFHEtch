#include "spdlog/spdlog.h"
#include "cpr/cpr.h"
#include "json/json.h"

#include "client_lib.h"

void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url("http://localhost:8080/ping"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);
}

void get_centroids() {
    SPDLOG_INFO("Sending a request to /query at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url("http://localhost:8080/query"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);
}

