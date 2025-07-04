#include <map>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

#include "client_server_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";

// Test method to ping server
void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at {}", server_addr);
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
    SPDLOG_INFO("Sending a request to /query at {}", server_addr);
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));
    // SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);

    const nlohmann::json resp = nlohmann::json::parse(r.text);
    centroids =
        resp.get<std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>>>();
    // SPDLOG_INFO("Retrieved centroids -> {}", resp.dump());
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

void get_coarse_scores(
    std::map<float, int64_t> &sorted_centroids,
    // Sending precise query temporarily, will be sending coarse vector in a
    // future implementation
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::vector<float> &coarse_scores,
    std::vector<faiss_idx_t> &coarse_vectors_idx,
    std::array<size_t, NQUERY> &list_sizes_per_query) {
    SPDLOG_INFO("Sending a request to /coarsesearch at {}", server_addr);

    std::array<int64_t, NPROBE> nearest_centroids_id;
    std::map<float, int64_t>::iterator nearest_centroids_it =
        sorted_centroids.begin();

    for (int i = 0;
         i < NPROBE && nearest_centroids_it != sorted_centroids.end(); i++) {
        nearest_centroids_id[i] = nearest_centroids_it->second;
        std::advance(nearest_centroids_it, 1);
    }

    nlohmann::json coarse_search_json;
    coarse_search_json["preciseQuery"] = precise_query;
    coarse_search_json["nearestCentroidIndexes"] = nearest_centroids_id;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}", resp.dump());

    coarse_scores = resp.at("coarseDistanceScores").get<std::vector<float>>();
    coarse_vectors_idx =
        resp.at("coarseVectorIndexes").get<std::vector<faiss_idx_t>>();
    list_sizes_per_query =
        resp.at("listSizesPerQuery").get<std::array<size_t, NQUERY>>();
}
