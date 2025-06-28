#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

// Include controllers headers to register with server
#include "controllers/Query.h"
#include "server_lib.h"

void init_logger() {}

void run_server() {
    init_logger();
    drogon::app().addListener("localhost", 8080);

    SPDLOG_INFO("Server listening on localhost:8080");
    drogon::app().run();
}

// Returns a dummy set of NUM_CENTROIDS centroids between 0 and 1
void retrieve_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids) {
    centroids.reserve(NUM_CENTROIDS);

    for (int i = 0; i < NUM_CENTROIDS; i++) {
        std::array<float, PRECISE_VECTOR_DIMENSIONS> centroid;

        float start = 0.0;
        constexpr float step = 1.0 / PRECISE_VECTOR_DIMENSIONS;
        for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++, start += step) {
            centroid[j] = start;
        }

        centroids.push_back(centroid);
    }
}