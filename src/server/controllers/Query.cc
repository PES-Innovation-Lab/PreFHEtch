#include <memory>
#include <vector>

#include <seal/seal.h>
#include <spdlog/spdlog.h>

#include "Query.h"
#include "client_server_utils.h"
#include "server_lib.h"
#include "prefhetch.pb.h"
#include "protobuf_utils.h"

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    Timer query_handler_timer;
    Timer retrieve_centroids_timer;
    Timer serde_parms_timer;

    SPDLOG_INFO("\n\nReceived request on /query");
    query_handler_timer.StartTimer();

    std::shared_ptr<Server> srvr = Server::getInstance();

    prefhetch::QueryResponse response;
    prefhetch_utils::vector_to_repeated(srvr->retrieve_centroids(), response.mutable_centroids());

    // serialise_parms() returns std::vector<seal::seal_byte>
    const auto& parms = srvr->serialise_parms();
    response.set_encrypted_parms(parms.data(), parms.size());
    response.set_subquantizers(srvr->SubQuantizers);

    std::string serialized_response;
    auto start_ser = std::chrono::high_resolution_clock::now();
    response.SerializeToString(&serialized_response);
    auto end_ser = std::chrono::high_resolution_clock::now();
    auto ser_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ser - start_ser).count();
    SPDLOG_INFO("QueryResponse serialization size: {} bytes, time: {} us", serialized_response.size(), ser_time);

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/x-protobuf");
    resp->setBody(serialized_response);

    callback(resp);
    query_handler_timer.StopTimer();

    SPDLOG_INFO("Time to retrieve centroids = {}(us)",
                retrieve_centroids_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to serialise encrypted params = {}(us)",
                serde_parms_timer.getDurationMicroseconds());
    SPDLOG_INFO("Exiting from query handler, total handler time = {}(us)",
                query_handler_timer.getDurationMicroseconds());
}

void Query::coarseSearch(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) {
    Timer coarse_search_handler_timer;
    Timer serde_coarse_search_params_timer;
    Timer coarse_search_timer;
    Timer serde_coarse_search_results_timer;

    SPDLOG_INFO("Received request on /coarsesearch");
    coarse_search_handler_timer.StartTimer();

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
        if (!server) {
            SPDLOG_ERROR("Failed to get Server instance");
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k500InternalServerError);
            callback(resp);
            return;
        }

        // Deserialize parameters
        serde_coarse_search_params_timer.StartTimer();
        auto [encrypted_residual_vectors, encrypted_residual_vectors_squared,
              relin_keys, galois_keys] = server->deserialise_coarse_search_parms(
            serde_residual_vecs,
            serde_residual_vecs_squared,
            serde_relin_keys,
            serde_galois_keys,
            serde_sk
        );
        serde_coarse_search_params_timer.StopTimer();

    coarse_search_timer.StartTimer();
    auto [encrypted_coarse_distances, coarse_vector_labels] =
        server->coarseSearch(nprobe_centroids, encrypted_residual_vectors,
                             encrypted_residual_vectors_squared, num_queries,
                             nprobe, relin_keys, galois_keys);
    coarse_search_timer.StopTimer();

    serde_coarse_search_results_timer.StartTimer();
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_encrypted_coarse_distances =
            server->serialise_encrypted_distances(encrypted_coarse_distances);
    serde_coarse_search_results_timer.StopTimer();

    nlohmann::json response;
    response["encryptedCoarseDistances"] = serde_encrypted_coarse_distances;
    response["coarseVectorLabels"] = coarse_vector_labels;
    SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
                getSizeInMB(response.dump().size()));

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    coarse_search_handler_timer.StopTimer();

    SPDLOG_INFO("Time to deserialise coarse search params = {}(us)",
                serde_coarse_search_params_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to perform coarse search = {}(us)",
                coarse_search_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to serialise coarse search results = {}(us)",
                serde_coarse_search_results_timer.getDurationMicroseconds());
    SPDLOG_INFO(
        "Size of the unserialised encrypted data = {}(mb)",
        getSizeInMB(getTotalNestedVecSize(serde_encrypted_coarse_distances)));
    SPDLOG_INFO(
        "Exiting from coarse search handler, total handler time = {}(us)",
        coarse_search_handler_timer.getDurationMicroseconds());
}

void Query::precise_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {

    Timer precise_search_handler_timer;
    Timer deserialise_precise_search_params_timer;
    Timer precise_search_timer;
    Timer serialise_precise_search_results_timer;

    SPDLOG_INFO("Received request on /precisesearch");
    precise_search_handler_timer.StartTimer();

    nlohmann::json req_body = nlohmann::json::parse(req->body());

    auto serde_encrypted_precise_queries =
        req_body.at("encryptedQueries")
            .get<std::vector<std::vector<seal::seal_byte>>>();
    auto nearest_coarse_vectors_id =
        req_body.at("nearestCoarseVectorsID")
            .get<std::vector<std::vector<faiss_idx_t>>>();
    auto serde_relin_keys =
        req_body["relinKeys"].get<std::vector<seal::seal_byte>>();
    auto serde_galois_keys =
        req_body["galoisKeys"].get<std::vector<seal::seal_byte>>();

    std::shared_ptr<Server> server = Server::getInstance();

    deserialise_precise_search_params_timer.StartTimer();
    auto [encrypted_precise_queries, relin_keys, galois_keys] =
        server->deserialise_precise_search_params(
            serde_encrypted_precise_queries, serde_relin_keys,
            serde_galois_keys);
    deserialise_precise_search_params_timer.StopTimer();

    precise_search_timer.StartTimer();
    std::vector<std::vector<seal::Ciphertext>> encrypted_precise_distances =
        server->preciseSearch(nearest_coarse_vectors_id,
                              encrypted_precise_queries, relin_keys,
                              galois_keys);
    precise_search_timer.StopTimer();

    serialise_precise_search_results_timer.StartTimer();
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_precise_search_results =
            server->serialise_encrypted_distances(encrypted_precise_distances);
    serialise_precise_search_results_timer.StopTimer();

    nlohmann::json response;
    response["encryptedPreciseDistances"] = serde_precise_search_results;
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    precise_search_handler_timer.StopTimer();

    SPDLOG_INFO(
        "Time to deserialise precise search params = {}(us)",
        deserialise_precise_search_params_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to perform precise search = {}(us)",
                precise_search_timer.getDurationMicroseconds());
    SPDLOG_INFO(
        "Time to serialise precise search results = {}(us)",
        serialise_precise_search_results_timer.getDurationMicroseconds());
    SPDLOG_INFO(
        "Exiting from precise search handler, total handler time = {}(us)",
        precise_search_handler_timer.getDurationMicroseconds());
}

void Query::precise_vector_pir(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /precise-vector-pir");

    // nlohmann::json req_body = nlohmann::json::parse(req->body());
    //
    // const std::array<std::array<faiss_idx_t, K>, NQUERY>
    //     k_nearest_precise_vectors_id =
    //         req_body.at("nearestPreciseVectorIndexes")
    //             .get<std::array<std::array<faiss_idx_t, K>, NQUERY>>();
    //
    // std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
    //            NQUERY>
    //     query_results;
    //
    // std::shared_ptr<Server> server = Server::getInstance();
    // server->preciseVectorPIR(k_nearest_precise_vectors_id, query_results);
    //
    nlohmann::json response;
    // response["queryResults"] = query_results;
    response["implement"] = "to be implemented";
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());

    callback(resp);
    SPDLOG_INFO("Exiting from precise vector PIR handler");
}

void Query::single_phase_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {

    Timer single_phase_search_handler_timer;
    Timer serde_search_params_timer;
    Timer encrypted_search_timer;
    Timer serde_search_results_timer;

    SPDLOG_INFO("Received request on /single-phase-search");
    single_phase_search_handler_timer.StartTimer();

    nlohmann::json req_body = nlohmann::json::parse(req->body());
    size_t num_queries = req_body["numQueries"].get<size_t>();
    auto serde_encrypted_query_vectors =
        req_body["encryptedQueries"]
            .get<std::vector<std::vector<seal::seal_byte>>>();
    auto nprobe_centroids =
        req_body["nearestCentroids"].get<std::vector<faiss::idx_t>>();
    auto serde_relin_keys =
        req_body["relinKeys"].get<std::vector<seal::seal_byte>>();
    auto serde_galois_keys =
        req_body["galoisKeys"].get<std::vector<seal::seal_byte>>();

    size_t nprobe = nprobe_centroids.size() / num_queries;

    std::shared_ptr<Server> server = Server::getInstance();

    serde_search_params_timer.StartTimer();
    auto [encrypted_query_vectors, relin_keys, galois_keys] =
        server->deserialise_single_phase_search_parms(
            serde_encrypted_query_vectors, serde_relin_keys, serde_galois_keys);
    serde_search_params_timer.StopTimer();

    encrypted_search_timer.StartTimer();
    auto [encrypted_coarse_distances, coarse_vector_labels] =
        server->singlePhaseSearch(nprobe_centroids, encrypted_query_vectors,
                                  num_queries, nprobe, relin_keys, galois_keys);
    encrypted_search_timer.StopTimer();

    serde_search_results_timer.StartTimer();
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_encrypted_single_phase_distances =
            server->serialise_encrypted_distances(encrypted_coarse_distances);
    serde_search_results_timer.StopTimer();

    nlohmann::json response;
    response["encryptedDistances"] = serde_encrypted_single_phase_distances;
    response["vectorLabels"] = coarse_vector_labels;
    SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
                getSizeInMB(response.dump().size()));

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(response.dump());
        // Prepare response
        prefhetch::CoarseSearchResponse response;
       
        // Serialize encrypted distances
        serde_coarse_search_results_timer.StartTimer();
        for (const auto& query_distances : encrypted_coarse_distances) {
            for (const auto& ciphertext : query_distances) {
                const size_t save_size = ciphertext.save_size();
                if (save_size == 0) {
                    SPDLOG_ERROR("Invalid ciphertext: save_size is 0");
                    auto resp = HttpResponse::newHttpResponse();
                    resp->setStatusCode(k500InternalServerError);
                    callback(resp);
                    return;
                }
                
                std::vector<seal::seal_byte> byte_vec(save_size);
                const size_t actual_size = ciphertext.save(
                    byte_vec.data(), 
                    byte_vec.size()
                );
                
                if (actual_size == 0 || actual_size > save_size) {
                    SPDLOG_ERROR("Ciphertext save failed: actual_size = {}, expected <= {}", 
                               actual_size, save_size);
                    auto resp = HttpResponse::newHttpResponse();
                    resp->setStatusCode(k500InternalServerError);
                    callback(resp);
                    return;
                }
                
                byte_vec.resize(actual_size);
                response.add_encrypted_coarse_distances(
                    byte_vec.data(), 
                    byte_vec.size()
                );
            }
        }
        
        // Add labels
        for (const auto& label_vec : coarse_vector_labels) {
            for (const auto& label : label_vec) {
                response.add_coarse_vector_labels(static_cast<uint64_t>(label));
            }
        }
        serde_coarse_search_results_timer.StopTimer();
        // Serialize and send response
        auto start_ser = std::chrono::high_resolution_clock::now();
        std::string serialized_response;
        if (!response.SerializeToString(&serialized_response)) {
            SPDLOG_ERROR("Failed to serialize CoarseSearchResponse");
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k500InternalServerError);
            callback(resp);
            return;
        }
        auto end_ser = std::chrono::high_resolution_clock::now();
        auto ser_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ser - start_ser).count();
        SPDLOG_INFO("CoarseSearchResponse serialization size: {} bytes, time: {} us", serialized_response.size(), ser_time);
        
        const HttpResponsePtr resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k200OK);
        resp->setContentTypeString("application/x-protobuf");
        resp->setBody(std::move(serialized_response));

    callback(resp);
    single_phase_search_handler_timer.StopTimer();

    SPDLOG_INFO("Time to deserialise single phase search params = {}(us)",
                serde_search_params_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to perform single phase search = {}(us)",
                encrypted_search_timer.getDurationMicroseconds());
    SPDLOG_INFO("Time to serialise single phase search results = {}(us)",
                serde_search_results_timer.getDurationMicroseconds());
    SPDLOG_INFO("Size of the unserialised encrypted data = {}(mb)",
                getSizeInMB(getTotalNestedVecSize(
                    serde_encrypted_single_phase_distances)));
    SPDLOG_INFO(
        "Exiting from single phase search handler, total handler time = {}(us)",
        single_phase_search_handler_timer.getDurationMicroseconds());
}
