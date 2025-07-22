#include <memory>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "Query.h"
#include "client_server_utils.h"
#include "server_lib.h"

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    Timer query_handler_timer;
    Timer retrieve_centroids_timer;
    Timer serde_parms_timer;

    SPDLOG_INFO("Received request on /query");
    query_handler_timer.StartTimer();

    std::shared_ptr<Server> srvr = Server::getInstance();

    nlohmann::json centroids_json;
    retrieve_centroids_timer.StartTimer();
    centroids_json["centroids"] = srvr->retrieve_centroids();
    retrieve_centroids_timer.StopTimer();

    serde_parms_timer.StartTimer();
    centroids_json["encryptedParms"] = srvr->serialise_parms();
    serde_parms_timer.StopTimer();

    centroids_json["subquantizers"] = srvr->SubQuantizers;

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(centroids_json.dump());

    callback(resp);
    query_handler_timer.StopTimer();

    SPDLOG_INFO("Retrieve centroids(microseconds) = {}",
                retrieve_centroids_timer.getDurationMicroseconds());
    SPDLOG_INFO("Serialise parms(microseconds) = {}",
                serde_parms_timer.getDurationMicroseconds());
    SPDLOG_INFO("Exiting from query handler, time(microseconds) = {}",
                query_handler_timer.getDurationMicroseconds());
}

void Query::coarse_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    // SPDLOG_INFO("Received request on /coarsesearch");

    nlohmann::json req_body = nlohmann::json::parse(req->body());
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        precise_query =
            req_body.at("preciseQuery")
                .get<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>,
                                NQUERY>>();
    std::array<std::array<faiss_idx_t, NPROBE>, NQUERY> nearest_centroids =
        req_body.at("nearestCentroidIndexes")
            .get<std::array<std::array<faiss_idx_t, NPROBE>, NQUERY>>();

    std::vector<float> coarse_distance_scores;
    std::vector<faiss::idx_t> coarse_vector_indexes;
    std::array<size_t, NQUERY> list_sizes_per_query;

    std::shared_ptr<Server> server = Server::getInstance();
    server->coarseSearch(precise_query, nearest_centroids,
                         coarse_distance_scores, coarse_vector_indexes,
                         list_sizes_per_query);

    nlohmann::json response;
    response["coarseDistanceScores"] = coarse_distance_scores;
    response["coarseVectorIndexes"] = coarse_vector_indexes;
    response["listSizesPerQuery"] = list_sizes_per_query;
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    // SPDLOG_INFO("Exiting from coarse search handler");
}

void Query::precise_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {

    // SPDLOG_INFO("Received request on /precisesearch");

    nlohmann::json req_body = nlohmann::json::parse(req->body());

    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        precise_query =
            req_body.at("preciseQuery")
                .get<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>,
                                NQUERY>>();
    const std::array<std::array<faiss_idx_t, COARSE_PROBE>, NQUERY>
        nearest_coarse_vectors_id =
            req_body.at("nearestCoarseVectorIndexes")
                .get<std::array<std::array<faiss_idx_t, COARSE_PROBE>,
                                NQUERY>>();

    std::array<std::array<float, COARSE_PROBE>, NQUERY> precise_distance_scores;

    std::shared_ptr<Server> server = Server::getInstance();
    server->preciseSearch(precise_query, nearest_coarse_vectors_id,
                          precise_distance_scores);

    nlohmann::json response;
    response["preciseDistanceScores"] = precise_distance_scores;
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    // SPDLOG_INFO("Exiting from precise search handler");
}

void Query::precise_vector_pir(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    // SPDLOG_INFO("Received request on /precise-vector-pir");

    nlohmann::json req_body = nlohmann::json::parse(req->body());

    const std::array<std::array<faiss_idx_t, K>, NQUERY>
        k_nearest_precise_vectors_id =
            req_body.at("nearestPreciseVectorIndexes")
                .get<std::array<std::array<faiss_idx_t, K>, NQUERY>>();

    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY>
        query_results;

    std::shared_ptr<Server> server = Server::getInstance();
    server->preciseVectorPIR(k_nearest_precise_vectors_id, query_results);

    nlohmann::json response;
    response["queryResults"] = query_results;
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    // SPDLOG_INFO("Exiting from precise vector PIR handler");
}
