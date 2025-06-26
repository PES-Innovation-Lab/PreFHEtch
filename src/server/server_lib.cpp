#include <vector>

#include "drogon/HttpAppFramework.h"
#include "spdlog/spdlog.h"
#include "json/json.h"

// Include controllers headers to register with server
#include "controllers/Query.h"
#include "server_lib.h"

void init_logger() {
}

void run_server() {
    init_logger();
    drogon::app().addListener("localhost", 8080);

    SPDLOG_INFO("Server listening on localhost:8080:");
    drogon::app().run();
}

// Returns a dummy set of NUM_CENTROIDS centroids between 0 and 1
std::vector<float> get_centroids() {
#define NUM_CENTROIDS 1000
    std::vector<float> centroids;
    centroids.reserve(NUM_CENTROIDS);

    float start = 0.0;
    float step = 1.0/NUM_CENTROIDS;
    for (int i=0; i<NUM_CENTROIDS; i++, start+=step) {
        centroids.push_back(start);
    }

    return centroids;
}

Json::Value get_centroids_json(const std::vector<float>& centroids) {
    Json::Value centroids_json;
    std::ostringstream oss;

    if (!centroids.empty()) {
        std::copy(centroids.begin(), centroids.end(), std::ostream_iterator<float>(oss, ","));
        oss << centroids.back();
    }

    centroids_json["centroids"] = oss.str();

    return centroids_json;
}
