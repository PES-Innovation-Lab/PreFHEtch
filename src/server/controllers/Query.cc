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

// void Query::coarseSearch(
//     const HttpRequestPtr &req,
//     std::function<void(const HttpResponsePtr &)> &&callback) {
//     Timer coarse_search_handler_timer;
//     Timer serde_coarse_search_params_timer;
//     Timer coarse_search_timer;
//     Timer serde_coarse_search_results_timer;
// 
//     SPDLOG_INFO("Received request on /coarsesearch");
//     coarse_search_handler_timer.StartTimer();
// 
//     nlohmann::json req_body = nlohmann::json::parse(req->body());
//     size_t num_queries = req_body["numQueries"].get<size_t>();
//     auto serde_encrypted_residual_vectors =
//         req_body["residualVecs"]
//             .get<std::vector<std::vector<std::vector<seal::seal_byte>>>>();
//     auto serde_encrypted_residual_vectors_squared =
//         req_body["residualVecsSquared"]
//             .get<std::vector<std::vector<std::vector<seal::seal_byte>>>>();
//     auto nprobe_centroids =
//         req_body["nearestCentroids"].get<std::vector<faiss::idx_t>>();
//     auto serde_relin_keys =
//         req_body["relinKeys"].get<std::vector<seal::seal_byte>>();
//     auto serde_galois_keys =
//         req_body["galoisKeys"].get<std::vector<seal::seal_byte>>();
// 
//     auto serde_sk = req_body["sk"].get<std::vector<seal::seal_byte>>();
// 
//     size_t nprobe = nprobe_centroids.size() / num_queries;
// 
//         std::shared_ptr<Server> server = Server::getInstance();
//         if (!server) {
//             SPDLOG_ERROR("Failed to get Server instance");
//             auto resp = HttpResponse::newHttpResponse();
//             resp->setStatusCode(k500InternalServerError);
//             callback(resp);
//             return;
//         }
// 
//         // Deserialize parameters
//         serde_coarse_search_params_timer.StartTimer();
//         auto [encrypted_residual_vectors, encrypted_residual_vectors_squared,
//               relin_keys, galois_keys] = server->deserialise_coarse_search_parms(
//             serde_residual_vecs,
//             serde_residual_vecs_squared,
//             serde_relin_keys,
//             serde_galois_keys,
//             serde_sk
//         );
//         serde_coarse_search_params_timer.StopTimer();
// 
//     coarse_search_timer.StartTimer();
//     auto [encrypted_coarse_distances, coarse_vector_labels] =
//         server->coarseSearch(nprobe_centroids, encrypted_residual_vectors,
//                              encrypted_residual_vectors_squared, num_queries,
//                              nprobe, relin_keys, galois_keys);
//     coarse_search_timer.StopTimer();
// 
//     serde_coarse_search_results_timer.StartTimer();
//     std::vector<std::vector<std::vector<seal::seal_byte>>>
//         serde_encrypted_coarse_distances =
//             server->serialise_encrypted_distances(encrypted_coarse_distances);
//     serde_coarse_search_results_timer.StopTimer();
// 
//     nlohmann::json response;
//     response["encryptedCoarseDistances"] = serde_encrypted_coarse_distances;
//     response["coarseVectorLabels"] = coarse_vector_labels;
//     SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
//                 getSizeInMB(response.dump().size()));
// 
//     const HttpResponsePtr resp = HttpResponse::newHttpResponse();
//     resp->setContentTypeString("application/json");
//     resp->setBody(response.dump());
// 
//     callback(resp);
//     coarse_search_handler_timer.StopTimer();
// 
//     SPDLOG_INFO("Time to deserialise coarse search params = {}(us)",
//                 serde_coarse_search_params_timer.getDurationMicroseconds());
//     SPDLOG_INFO("Time to perform coarse search = {}(us)",
//                 coarse_search_timer.getDurationMicroseconds());
//     SPDLOG_INFO("Time to serialise coarse search results = {}(us)",
//                 serde_coarse_search_results_timer.getDurationMicroseconds());
//     SPDLOG_INFO(
//         "Size of the unserialised encrypted data = {}(mb)",
//         getSizeInMB(getTotalNestedVecSize(serde_encrypted_coarse_distances)));
//     SPDLOG_INFO(
//         "Exiting from coarse search handler, total handler time = {}(us)",
//         coarse_search_handler_timer.getDurationMicroseconds());
// }

// At the top of the file, make sure you have the right includes
#include "server/controllers/Query.h"
#include "server/server_lib.h"
#include "utils/timer.h"
#include "spdlog/spdlog.h"
// Remove any nlohmann/json includes since we're using protobuf only

void Query::coarseSearch(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) {
    Timer timer;
    SPDLOG_INFO("Received request on /coarsesearch");
    timer.StartTimer();

    try {
        // Parse protobuf request
        prefhetch::CoarseSearchRequest request;
        if (!request.ParseFromString(req->body())) {
            SPDLOG_ERROR("Failed to parse CoarseSearchRequest");
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        const size_t num_queries = request.num_queries();
        if (num_queries != static_cast<size_t>(request.queries_size())) {
            SPDLOG_ERROR("Query count mismatch: expected {}, got {}", 
                        num_queries, request.queries_size());
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k400BadRequest);
            callback(resp);
            return;
        }

        SPDLOG_INFO("Processing {} queries", num_queries);

        // Extract data from protobuf
        std::vector<faiss::idx_t> nprobe_centroids;
        std::vector<std::vector<std::vector<seal::seal_byte>>> serde_residual_vecs;
        std::vector<std::vector<std::vector<seal::seal_byte>>> serde_residual_vecs_squared;
        
        size_t total_residuals = 0;
        size_t total_residuals_squared = 0;
        size_t total_centroids = 0;
        
        for (const auto& query : request.queries()) {
            // Residual vectors
            std::vector<std::vector<seal::seal_byte>> query_residuals;
            query_residuals.reserve(query.residual_vecs_size());
            
            for (const auto& bytes : query.residual_vecs()) {
                std::vector<seal::seal_byte> vec;
                vec.reserve(bytes.size());
                for (char c : bytes) {
                    vec.push_back(static_cast<seal::seal_byte>(c));
                }
                query_residuals.push_back(std::move(vec));
                total_residuals++;
            }
            serde_residual_vecs.push_back(std::move(query_residuals));
            
            // Residual vectors squared
            std::vector<std::vector<seal::seal_byte>> query_residuals_squared;
            query_residuals_squared.reserve(query.residual_vecs_squared_size());
            
            for (const auto& bytes : query.residual_vecs_squared()) {
                std::vector<seal::seal_byte> vec;
                vec.reserve(bytes.size());
                for (char c : bytes) {
                    vec.push_back(static_cast<seal::seal_byte>(c));
                }
                query_residuals_squared.push_back(std::move(vec));
                total_residuals_squared++;
            }
            serde_residual_vecs_squared.push_back(std::move(query_residuals_squared));
            
            // Centroids
            for (auto centroid : query.nearest_centroids()) {
                nprobe_centroids.push_back(static_cast<faiss::idx_t>(centroid));
                total_centroids++;
            }
        }

        SPDLOG_INFO("Extracted: residuals={}, residuals_squared={}, centroids={}", 
                   total_residuals, total_residuals_squared, total_centroids);

        // Keys - Convert string to seal::seal_byte vector
        std::vector<seal::seal_byte> serde_relin_keys;
        serde_relin_keys.reserve(request.relin_keys().size());
        for (char c : request.relin_keys()) {
            serde_relin_keys.push_back(static_cast<seal::seal_byte>(c));
        }

        std::vector<seal::seal_byte> serde_galois_keys;
        serde_galois_keys.reserve(request.galois_keys().size());
        for (char c : request.galois_keys()) {
            serde_galois_keys.push_back(static_cast<seal::seal_byte>(c));
        }

        std::vector<seal::seal_byte> serde_sk;
        serde_sk.reserve(request.sk().size());
        for (char c : request.sk()) {
            serde_sk.push_back(static_cast<seal::seal_byte>(c));
        }

        SPDLOG_INFO("Key sizes: relin={}, galois={}, sk={}", 
                   serde_relin_keys.size(), serde_galois_keys.size(), serde_sk.size());

        // Get SEAL context (you'll need to get this from your server instance)
        auto server = Server::getInstance();
        seal::SEALContext seal_ctx = server->getSEALContext(); // Assuming this method exists
        
        // Deserialize keys
        seal::RelinKeys relin_keys;
        relin_keys.load(seal_ctx, serde_relin_keys.data(), serde_relin_keys.size());
        
        seal::GaloisKeys galois_keys; 
        galois_keys.load(seal_ctx, serde_galois_keys.data(), serde_galois_keys.size());
        
        // Convert serialized ciphertexts to actual Ciphertext objects
        std::vector<std::vector<seal::Ciphertext>> encrypted_residual_queries;
        std::vector<std::vector<seal::Ciphertext>> encrypted_residual_queries_squared;
        
        encrypted_residual_queries.reserve(serde_residual_vecs.size());
        for (const auto& query_vecs : serde_residual_vecs) {
            std::vector<seal::Ciphertext> query_ciphertexts;
            query_ciphertexts.reserve(query_vecs.size());
            for (const auto& vec : query_vecs) {
                seal::Ciphertext ct;
                ct.load(seal_ctx, vec.data(), vec.size());
                query_ciphertexts.push_back(std::move(ct));
            }
            encrypted_residual_queries.push_back(std::move(query_ciphertexts));
        }
        
        encrypted_residual_queries_squared.reserve(serde_residual_vecs_squared.size());
        for (const auto& query_vecs : serde_residual_vecs_squared) {
            std::vector<seal::Ciphertext> query_ciphertexts;
            query_ciphertexts.reserve(query_vecs.size());
            for (const auto& vec : query_vecs) {
                seal::Ciphertext ct;
                ct.load(seal_ctx, vec.data(), vec.size());
                query_ciphertexts.push_back(std::move(ct));
            }
            encrypted_residual_queries_squared.push_back(std::move(query_ciphertexts));
        }
        
        // Calculate nprobe from centroids per query
        size_t nprobe = total_centroids / num_queries;
        
        SPDLOG_INFO("Calling server->coarseSearch with {} queries, nprobe={}", num_queries, nprobe);
        
        // Process with server - now with correct 7 arguments
        auto [encrypted_coarse_distances, coarse_vector_labels] =
            server->coarseSearch(nprobe_centroids, encrypted_residual_queries, 
                               encrypted_residual_queries_squared, num_queries, 
                               nprobe, relin_keys, galois_keys);

        // Build response
        prefhetch::CoarseSearchResponse response;
        response.mutable_results()->Reserve(encrypted_coarse_distances.size());
        
        for (size_t i = 0; i < encrypted_coarse_distances.size(); i++) {
            auto* result = response.add_results();
            
            // Reserve space for efficiency
            result->mutable_encrypted_coarse_distances()->Reserve(encrypted_coarse_distances[i].size());
            result->mutable_coarse_vector_labels()->Reserve(coarse_vector_labels[i].size());
            
            // Distances - need to serialize Ciphertext objects back to bytes
            for (const auto& ciphertext : encrypted_coarse_distances[i]) {
                std::vector<seal::seal_byte> bytes(ciphertext.save_size());
                ciphertext.save(bytes.data(), bytes.size());
                result->add_encrypted_coarse_distances()->assign(
                    reinterpret_cast<const char*>(bytes.data()), 
                    bytes.size()
                );
            }
            
            // Labels
            for (auto label : coarse_vector_labels[i]) {
                result->add_coarse_vector_labels(static_cast<uint64_t>(label));
            }
        }

        // Serialize response with timing
        auto start_ser = std::chrono::high_resolution_clock::now();
        std::string serialized;
        if (!response.SerializeToString(&serialized)) {
            throw std::runtime_error("Failed to serialize response");
        }
        auto end_ser = std::chrono::high_resolution_clock::now();
        auto ser_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ser - start_ser).count();
        
        SPDLOG_INFO("Response serialization: size={} bytes, time={} us", 
                   serialized.size(), ser_time);
        
        auto http_resp = HttpResponse::newHttpResponse();
        http_resp->setBody(serialized);
        http_resp->addHeader("Content-Type", "application/x-protobuf");
        callback(http_resp);
        
        timer.StopTimer();
        // Use the correct timer method name (check your Timer class implementation)
        SPDLOG_INFO("Total coarseSearch processing time: {} ms", timer.GetElapsedTime());
        
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Error in coarseSearch: {}", e.what());
        auto resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k500InternalServerError);
        resp->setBody(std::string("Internal server error: ") + e.what());
        callback(resp);
    }
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
