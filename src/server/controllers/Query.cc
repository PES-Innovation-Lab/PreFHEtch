#include <memory>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <seal/decryptor.h>
#include <seal/encryptionparams.h>
#include <seal/secretkey.h>

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
    SPDLOG_INFO("Received request on /coarsesearch");

    Timer serde_coarse_search_params_timer;
    Timer coarse_search_timer;
    Timer serde_coarse_search_results_timer;

    nlohmann::json req_body = nlohmann::json::parse(req->body());
    size_t num_queries = req_body["numQueries"].get<size_t>();
    auto serde_encrypted_residual_vectors =
        req_body["residualVecs"]
            .get<std::vector<std::vector<std::vector<seal::seal_byte>>>>();
    auto serde_encrypted_residual_vectors_squared =
        req_body["residualVecsSquared"]
            .get<std::vector<std::vector<std::vector<seal::seal_byte>>>>();
    auto nprobe_centroids =
        req_body["nearestCentroids"].get<std::vector<faiss::idx_t>>();
    auto serde_relin_keys =
        req_body["relinKeys"].get<std::vector<seal::seal_byte>>();
    auto serde_galois_keys =
        req_body["galoisKeys"].get<std::vector<seal::seal_byte>>();

    auto serde_sk = req_body["sk"].get<std::vector<seal::seal_byte>>();

    size_t nprobe = nprobe_centroids.size() / num_queries;

    std::shared_ptr<Server> server = Server::getInstance();
    // server->display_nprobe_centroids(nprobe_centroids, num_queries);

    serde_coarse_search_params_timer.StartTimer();
    auto [encrypted_residual_vectors, encrypted_residual_vectors_squared,
          relin_keys, galois_keys] =
        server->deserialise_coarse_search_parms(
            serde_encrypted_residual_vectors,
            serde_encrypted_residual_vectors_squared, serde_relin_keys,
            serde_galois_keys, serde_sk);
    serde_coarse_search_params_timer.StopTimer();

    SPDLOG_INFO("Time to deserialise coarse search params = {}(microseconds)",
                serde_coarse_search_params_timer.getDurationMicroseconds());

    coarse_search_timer.StartTimer();
    auto [encrypted_coarse_distances, coarse_vector_labels] =
        server->coarseSearch(nprobe_centroids, encrypted_residual_vectors,
                             encrypted_residual_vectors_squared, num_queries,
                             nprobe, relin_keys, galois_keys, serde_sk);

    SPDLOG_INFO("Time to perform coarse search = {}(microseconds)",
                coarse_search_timer.getDurationMicroseconds());

    serde_coarse_search_results_timer.StartTimer();
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_encrypted_coarse_distances =
            server->serialise_encrypted_coarse_distances(
                encrypted_coarse_distances);
    serde_coarse_search_results_timer.StopTimer();

    SPDLOG_INFO("Size of the unserialised encrypted data = {}",
                getTotalSize(serde_encrypted_coarse_distances));

    SPDLOG_INFO("Time to serialise coarse search results = {}(microseconds)",
                serde_coarse_search_results_timer.getDurationMicroseconds());

    nlohmann::json response;
    response["encryptedCoarseDistances"] = serde_encrypted_coarse_distances;
    response["coarseVectorLabels"] = coarse_vector_labels;
    SPDLOG_INFO("Size of the serialised encrypted data = {}",
    response.dump().size());

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    SPDLOG_INFO("Exiting from coarse search handler");
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
