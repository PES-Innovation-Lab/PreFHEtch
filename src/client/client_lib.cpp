#include <iostream>
#include <map>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

#include "client_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";

// Test method to ping server
void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "ping"));
    const nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}, Status code = {}", resp.dump(), r.status_code);
}

void get_query(std::array<float, PRECISE_VECTOR_DIMENSIONS> &query) {
    size_t nq;
    std::vector<float> xq;

    SPDLOG_INFO("Loading query:");

    size_t d2;
    vecs_read<float>(QUERY_DATASET_PATH, d2, nq, xq);
    assert(PRECISE_VECTOR_DIMENSIONS == d2 ||
           !"query does not have same dimension as train set");

    for (int i = 0; i < PRECISE_VECTOR_DIMENSIONS; i++) {
        query[i] = xq[i];
        printf("%.1f, ", xq[i]);
    }
    printf("\n");
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

void sort_nearest_centroids(
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    const std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids,
    std::map<float, int64_t> &nearest_centroids_idx) {

    for (int i = 0; i < centroids.size(); i++) {
        float distance = 0.0;
        for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++) {
            distance += std::pow(precise_query[j] - centroids[i][j], 2);
        }
        nearest_centroids_idx[distance] = i;
    }
}
