#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

// Test method to ping server
void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "ping"));
    const nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}, Status code = {}", resp.dump(), r.status_code);
}

void get_query(std::vector<float> &precise_query) {
    precise_query.reserve(PRECISE_VECTOR_DIMENSIONS);

    float start = 0.0;
    constexpr float step = 1.0 / PRECISE_VECTOR_DIMENSIONS;
    for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++, start += step) {
        precise_query.push_back(start);
    }
}

void get_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids) {
    SPDLOG_INFO("Sending a request to /query at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);

    const nlohmann::json resp = nlohmann::json::parse(r.text);
    centroids =
        resp.get<std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>>>();
    SPDLOG_INFO("Retrieved centroids -> {}", resp.dump());
}
