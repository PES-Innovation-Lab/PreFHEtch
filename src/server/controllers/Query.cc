#include <memory>
#include <vector>
#include <cstring>

#include <seal/seal.h>
#include <spdlog/spdlog.h>

#include "Query.h"
#include "client_server_utils.h"
#include "server_lib.h"
#include "search.pb.h"

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    Timer query_handler_timer;
    Timer retrieve_centroids_timer;
    Timer serde_parms_timer;

    SPDLOG_INFO("\n\nReceived request on /query");
    query_handler_timer.StartTimer();

    std::shared_ptr<Server> srvr = Server::getInstance();

    prefhetch::proto::QueryResponse response;
    
    retrieve_centroids_timer.StartTimer();
    std::vector<float> centroids = srvr->retrieve_centroids();
    for (float centroid : centroids) {
        response.add_centroids(centroid);
    }
    retrieve_centroids_timer.StopTimer();

    serde_parms_timer.StartTimer();
    std::vector<seal::seal_byte> encrypted_parms = srvr->serialise_parms();
    response.set_encrypted_parms(encrypted_parms.data(), encrypted_parms.size());
    serde_parms_timer.StopTimer();

    response.set_subquantizers(srvr->SubQuantizers);

    std::string serialized_response;
    response.SerializeToString(&serialized_response);

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/x-protobuf");
    resp->setBody(serialized_response);

    callback(resp);
    query_handler_timer.StopTimer();

    SPDLOG_INFO("Time to retrieve centroids = {}(ms)",
                retrieve_centroids_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to serialise encrypted params = {}(ms)",
                serde_parms_timer.getDurationMilliseconds());
    SPDLOG_INFO("Exiting from query handler, total handler time = {}(ms)",
                query_handler_timer.getDurationMilliseconds());
}

void Query::coarse_search(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    Timer coarse_search_handler_timer;
    Timer serde_coarse_search_params_timer;
    Timer coarse_search_timer;
    Timer serde_coarse_search_results_timer;

    SPDLOG_INFO("Received request on /coarsesearch");
    coarse_search_handler_timer.StartTimer();

    prefhetch::proto::CoarseSearchRequest request;
    if (!request.ParseFromString(req->body())) {
        SPDLOG_ERROR("Failed to parse protobuf coarse search request");
        const HttpResponsePtr resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    size_t num_queries = request.num_queries();
    
    std::vector<std::vector<std::vector<seal::seal_byte>>> serde_encrypted_residual_vectors;
    serde_encrypted_residual_vectors.reserve(request.residual_vecs_size());
    
    for (const auto& query_vecs : request.residual_vecs()) {
        std::vector<std::vector<seal::seal_byte>> query_vectors;
        query_vectors.reserve(query_vecs.vectors_size());
        
        for (const auto& vec_data : query_vecs.vectors()) {
            std::vector<seal::seal_byte> vector_bytes;
            vector_bytes.resize(vec_data.size());
            std::memcpy(vector_bytes.data(), vec_data.data(), vec_data.size());
            query_vectors.push_back(std::move(vector_bytes));
        }
        serde_encrypted_residual_vectors.push_back(std::move(query_vectors));
    }
    
    std::vector<std::vector<std::vector<seal::seal_byte>>> serde_encrypted_residual_vectors_squared;
    serde_encrypted_residual_vectors_squared.reserve(request.residual_vecs_squared_size());
    
    for (const auto& query_vecs : request.residual_vecs_squared()) {
        std::vector<std::vector<seal::seal_byte>> query_vectors;
        query_vectors.reserve(query_vecs.vectors_size());
        
        for (const auto& vec_data : query_vecs.vectors()) {
            std::vector<seal::seal_byte> vector_bytes;
            vector_bytes.resize(vec_data.size());
            std::memcpy(vector_bytes.data(), vec_data.data(), vec_data.size());
            query_vectors.push_back(std::move(vector_bytes));
        }
        serde_encrypted_residual_vectors_squared.push_back(std::move(query_vectors));
    }
    
    std::vector<faiss::idx_t> nprobe_centroids;
    nprobe_centroids.reserve(request.nearest_centroids_size());
    for (int i = 0; i < request.nearest_centroids_size(); ++i) {
        nprobe_centroids.push_back(static_cast<faiss::idx_t>(request.nearest_centroids(i)));
    }
    
    std::vector<seal::seal_byte> serde_relin_keys;
    const std::string& relin_data = request.relin_keys();
    serde_relin_keys.resize(relin_data.size());
    std::memcpy(serde_relin_keys.data(), relin_data.data(), relin_data.size());
    
    std::vector<seal::seal_byte> serde_galois_keys;
    const std::string& galois_data = request.galois_keys();
    serde_galois_keys.resize(galois_data.size());
    std::memcpy(serde_galois_keys.data(), galois_data.data(), galois_data.size());
    
    std::vector<seal::seal_byte> serde_sk;
    const std::string& sk_data = request.sk();
    serde_sk.resize(sk_data.size());
    std::memcpy(serde_sk.data(), sk_data.data(), sk_data.size());

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

    prefhetch::proto::CoarseSearchResponse response;
    
    for (const auto& query_distances : serde_encrypted_coarse_distances) {
        auto* query_vecs = response.add_encrypted_coarse_distances();
        for (const auto& distance_bytes : query_distances) {
            query_vecs->add_vectors(distance_bytes.data(), distance_bytes.size());
        }
    }
    
    for (const auto& query_labels : coarse_vector_labels) {
        auto* labels = response.add_coarse_vector_labels();
        for (faiss::idx_t label : query_labels) {
            labels->add_labels(static_cast<int64_t>(label));
        }
    }

    std::string serialized_response;
    response.SerializeToString(&serialized_response);
    
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/x-protobuf");
    resp->setBody(serialized_response);

    callback(resp);
    coarse_search_handler_timer.StopTimer();

    SPDLOG_INFO("Time to deserialise coarse search params = {}(ms)",
                serde_coarse_search_params_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to perform coarse search = {}(ms)",
                coarse_search_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to serialise coarse search results = {}(ms)",
                serde_coarse_search_results_timer.getDurationMilliseconds());
    SPDLOG_INFO(
        "Size of the unserialised encrypted data = {}(mb)",
        getSizeInMB(getTotalNestedVecSize(serde_encrypted_coarse_distances)));
    SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
                getSizeInMB(serialized_response.size()));
    SPDLOG_INFO(
        "Exiting from coarse search handler, total handler time = {}(ms)",
        coarse_search_handler_timer.getDurationMilliseconds());
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

    prefhetch::proto::PreciseSearchRequest request;
    if (!request.ParseFromString(req->body())) {
        SPDLOG_ERROR("Failed to parse protobuf precise search request");
        const HttpResponsePtr resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    std::vector<std::vector<seal::seal_byte>> serde_encrypted_precise_queries;
    serde_encrypted_precise_queries.reserve(request.encrypted_queries_size());
    
    for (int i = 0; i < request.encrypted_queries_size(); ++i) {
        const std::string& query_data = request.encrypted_queries(i);
        std::vector<seal::seal_byte> query_bytes;
        query_bytes.resize(query_data.size());
        std::memcpy(query_bytes.data(), query_data.data(), query_data.size());
        serde_encrypted_precise_queries.push_back(std::move(query_bytes));
    }
    
    std::vector<std::vector<faiss_idx_t>> nearest_coarse_vectors_id;
    nearest_coarse_vectors_id.reserve(request.nearest_coarse_vectors_id_size());
    
    for (const auto& vector_labels : request.nearest_coarse_vectors_id()) {
        std::vector<faiss_idx_t> labels;
        labels.reserve(vector_labels.labels_size());
        
        for (int i = 0; i < vector_labels.labels_size(); ++i) {
            labels.push_back(static_cast<faiss_idx_t>(vector_labels.labels(i)));
        }
        nearest_coarse_vectors_id.push_back(std::move(labels));
    }
    
    std::vector<seal::seal_byte> serde_relin_keys;
    const std::string& relin_data = request.relin_keys();
    serde_relin_keys.resize(relin_data.size());
    std::memcpy(serde_relin_keys.data(), relin_data.data(), relin_data.size());
    
    std::vector<seal::seal_byte> serde_galois_keys;
    const std::string& galois_data = request.galois_keys();
    serde_galois_keys.resize(galois_data.size());
    std::memcpy(serde_galois_keys.data(), galois_data.data(), galois_data.size());

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

    prefhetch::proto::PreciseSearchResponse response;
    
    for (const auto& query_distances : serde_precise_search_results) {
        for (const auto& distance_bytes : query_distances) {
            response.add_encrypted_precise_distances(distance_bytes.data(), distance_bytes.size());
        }
    }

    std::string serialized_response;
    response.SerializeToString(&serialized_response);
    
    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/x-protobuf");
    resp->setBody(serialized_response);

    callback(resp);
    precise_search_handler_timer.StopTimer();

    SPDLOG_INFO(
        "Time to deserialise precise search params = {}(ms)",
        deserialise_precise_search_params_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to perform precise search = {}(ms)",
                precise_search_timer.getDurationMilliseconds());
    SPDLOG_INFO(
        "Time to serialise precise search results = {}(ms)",
        serialise_precise_search_results_timer.getDurationMilliseconds());
    SPDLOG_INFO(
        "Size of the unserialised encrypted data = {}(mb)",
        getSizeInMB(getTotalNestedVecSize(serde_precise_search_results)));
    SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
                getSizeInMB(serialized_response.size()));
    SPDLOG_INFO(
        "Exiting from precise search handler, total handler time = {}(ms)",
        precise_search_handler_timer.getDurationMilliseconds());
}

void Query::precise_vector_pir(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /precise-vector-pir");

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("text/plain");
    resp->setBody("to be implemented");

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

    prefhetch::proto::SinglePhaseSearchRequest request;
    if (!request.ParseFromString(req->body())) {
        SPDLOG_ERROR("Failed to parse protobuf single phase search request");
        const HttpResponsePtr resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k400BadRequest);
        callback(resp);
        return;
    }

    size_t num_queries = request.num_queries();
    
    std::vector<std::vector<seal::seal_byte>> serde_encrypted_query_vectors;
    serde_encrypted_query_vectors.reserve(request.encrypted_queries_size());
    
    for (int i = 0; i < request.encrypted_queries_size(); ++i) {
        const std::string& query_data = request.encrypted_queries(i);
        std::vector<seal::seal_byte> query_bytes;
        query_bytes.resize(query_data.size());
        std::memcpy(query_bytes.data(), query_data.data(), query_data.size());
        serde_encrypted_query_vectors.push_back(std::move(query_bytes));
    }
    
    std::vector<faiss::idx_t> nprobe_centroids;
    nprobe_centroids.reserve(request.nearest_centroids_size());
    for (int i = 0; i < request.nearest_centroids_size(); ++i) {
        nprobe_centroids.push_back(static_cast<faiss::idx_t>(request.nearest_centroids(i)));
    }
    
    std::vector<seal::seal_byte> serde_relin_keys;
    const std::string& relin_data = request.relin_keys();
    serde_relin_keys.resize(relin_data.size());
    std::memcpy(serde_relin_keys.data(), relin_data.data(), relin_data.size());
    
    std::vector<seal::seal_byte> serde_galois_keys;
    const std::string& galois_data = request.galois_keys();
    serde_galois_keys.resize(galois_data.size());
    std::memcpy(serde_galois_keys.data(), galois_data.data(), galois_data.size());

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

    prefhetch::proto::SinglePhaseSearchResponse response;
    
    for (const auto& query_distances : serde_encrypted_single_phase_distances) {
        auto* query_vecs = response.add_encrypted_distances();
        for (const auto& distance_bytes : query_distances) {
            query_vecs->add_vectors(distance_bytes.data(), distance_bytes.size());
        }
    }
    
    for (const auto& query_labels : coarse_vector_labels) {
        auto* labels = response.add_vector_labels();
        for (faiss::idx_t label : query_labels) {
            labels->add_labels(static_cast<int64_t>(label));
        }
    }

    std::string serialized_response;
    response.SerializeToString(&serialized_response);
    

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/x-protobuf");
    resp->setBody(serialized_response);

    callback(resp);
    single_phase_search_handler_timer.StopTimer();

    SPDLOG_INFO("Time to deserialise single phase search params = {}(ms)",
                serde_search_params_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to perform single phase search = {}(ms)",
                encrypted_search_timer.getDurationMilliseconds());
    SPDLOG_INFO("Time to serialise single phase search results = {}(ms)",
                serde_search_results_timer.getDurationMilliseconds());
    SPDLOG_INFO("Size of the unserialised encrypted data = {}(mb)",
                getSizeInMB(getTotalNestedVecSize(
                    serde_encrypted_single_phase_distances)));
    SPDLOG_INFO("Size of the serialised encrypted data = {}(mb)",
                getSizeInMB(serialized_response.size()));
    SPDLOG_INFO(
        "Exiting from single phase search handler, total handler time = {}(ms)",
        single_phase_search_handler_timer.getDurationMilliseconds());
}
