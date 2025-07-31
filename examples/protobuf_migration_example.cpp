// Example: Migrating from JSON to Protobuf
// This shows how to convert the get_centroids_encrypted_parms() function

#include <cpr/cpr.h>
#include "prefhetch.pb.h"
#include "protobuf_utils.h"

// OLD JSON VERSION:
/*
std::pair<std::vector<float>, std::vector<seal::seal_byte>>
Client::get_centroids_encrypted_parms() {
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));

    const nlohmann::json resp = nlohmann::json::parse(r.text);
    std::vector<float> centroids =
        resp.at("centroids").get<std::vector<float>>();
    std::vector<seal::seal_byte> encrypted_parms =
        resp.at("encryptedParms").get<std::vector<seal::seal_byte>>();
    m_Subquantizers = resp.at("subquantizers").get<size_t>();

    m_Nlist = centroids.size() / m_PreciseVectorDimensions;

    if (m_NProbe > m_Nlist) {
        SPDLOG_ERROR("NProbe is greater than Nlist");
        throw std::runtime_error("NProbe is greater than Nlist");
    }

    return {centroids, encrypted_parms};
}
*/

// NEW PROTOBUF VERSION:
std::pair<std::vector<float>, std::vector<seal::seal_byte>>
Client::get_centroids_encrypted_parms() {
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));

    // Parse protobuf message from response
    prefhetch::QueryResponse resp;
    if (!resp.ParseFromString(r.text)) {
        throw std::runtime_error("Failed to parse protobuf response");
    }

    // Extract data from protobuf message
    std::vector<float> centroids;
    prefhetch_utils::repeated_to_vector(resp.centroids(), centroids);
    
    std::vector<seal::seal_byte> encrypted_parms;
    prefhetch_utils::repeated_to_vector(resp.encrypted_parms(), encrypted_parms);
    
    m_Subquantizers = resp.subquantizers();

    m_Nlist = centroids.size() / m_PreciseVectorDimensions;

    if (m_NProbe > m_Nlist) {
        SPDLOG_ERROR("NProbe is greater than Nlist");
        throw std::runtime_error("NProbe is greater than Nlist");
    }

    return {centroids, encrypted_parms};
}

// Example: Migrating a request function (coarse search)
/*
// OLD JSON VERSION:
nlohmann::json coarse_search_json;
coarse_search_json["numQueries"] = m_NumQueries;
coarse_search_json["residualVecs"] = serde_encrypted_vecs;
coarse_search_json["residualVecsSquared"] = serde_encrypted_vecs_squared;
coarse_search_json["nearestCentroids"] = nprobe_nearest_centroids_idx;
coarse_search_json["relinKeys"] = serde_relin_keys;
coarse_search_json["galoisKeys"] = serde_galois_keys;
coarse_search_json["sk"] = serde_sk;

cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                            cpr::Body(coarse_search_json.dump()));
*/

// NEW PROTOBUF VERSION:
void Client::send_coarse_search_request() {
    prefhetch::CoarseSearchRequest request;
    
    request.set_num_queries(m_NumQueries);
    
    // Convert vectors to protobuf repeated fields
    prefhetch_utils::vector_to_repeated(serde_encrypted_vecs, request.mutable_residual_vecs());
    prefhetch_utils::vector_to_repeated(serde_encrypted_vecs_squared, request.mutable_residual_vecs_squared());
    prefhetch_utils::vector_to_repeated(nprobe_nearest_centroids_idx, request.mutable_nearest_centroids());
    prefhetch_utils::vector_to_repeated(serde_relin_keys, request.mutable_relin_keys());
    prefhetch_utils::vector_to_repeated(serde_galois_keys, request.mutable_galois_keys());
    prefhetch_utils::vector_to_repeated(serde_sk, request.mutable_sk());

    // Serialize to string
    std::string serialized_request;
    if (!request.SerializeToString(&serialized_request)) {
        throw std::runtime_error("Failed to serialize protobuf request");
    }

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(serialized_request));
    
    // Parse response
    prefhetch::CoarseSearchResponse response;
    if (!response.ParseFromString(r.text)) {
        throw std::runtime_error("Failed to parse protobuf response");
    }
    
    // Use response.response() instead of resp["response"]
} 