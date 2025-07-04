#include <memory>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "Query.h"
#include "server_lib.h"
#include "client_server_utils.h"

void Query::ping(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /ping");
    nlohmann::json ret;
    ret["ping"] = "pong";

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(ret.dump());

    callback(resp);
}

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /query");

    std::shared_ptr<Server> srvr = Server::getInstance();

    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> centroids;
    srvr->retrieve_centroids(centroids);
    const nlohmann::json centroids_json = centroids;

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(centroids_json.dump());

    callback(resp);
}

void Query::coarse_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /coarsesearch");

    nlohmann::json req_body = nlohmann::json::parse(req->body());
    SPDLOG_INFO("Request body: {}", req_body.dump());

    const std::array<float, PRECISE_VECTOR_DIMENSIONS> precise_query =
        req_body["preciseQuery"];
    std::array<int64_t, NPROBE> nearest_centroids =
        req_body["nearestCentroidIndexes"];
    std::vector<float> coarse_distance_scores;
    std::vector<faiss::idx_t> coarse_vector_indexes;
    std::array<size_t, NQUERY> list_sizes_per_query;

    std::shared_ptr<Server> server = Server::getInstance();
    server->prefilter(precise_query, nearest_centroids, coarse_distance_scores,
                      coarse_vector_indexes, list_sizes_per_query);

    nlohmann::json response;
    response["coarseDistanceScores"] = coarse_distance_scores;
    response["coarseVectorIndexes"] = coarse_vector_indexes;
    response["listSizesPerQuery"] = list_sizes_per_query;
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
}
