#include <algorithm>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"
#include "client_server_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

Encryption::Encryption(seal::EncryptionParameters encrypt_parms,
                       const seal::SEALContext &seal_ctx)
    : EncryptedParms(std::move(encrypt_parms)), KeyGen(seal_ctx),
      SecretKey(KeyGen.secret_key()),
      SerdeRelinKeys(KeyGen.create_relin_keys()),
      SerdeGaloisKeys(KeyGen.create_galois_keys()),
      Encryptor(seal_ctx, SecretKey), Decryptor(seal_ctx, SecretKey),
      BatchEncoder(seal_ctx) {}

Client::Client(size_t num_queries, size_t nprobe) {
    m_NumQueries = num_queries;
    m_NProbe = nprobe;
}

std::vector<float> Client::get_query() {
    size_t parsed_num_queries;
    std::vector<float> parsed_precise_queries;

    vecs_read<float>(QUERY_DATASET_PATH, m_PreciseVectorDimensions,
                     parsed_num_queries, parsed_precise_queries);

    if (m_NumQueries > parsed_num_queries) {
        SPDLOG_ERROR("insufficient queries present in dataset");
        throw std::runtime_error("insufficient queries present in dataset");
    }

    // for (const float &ele : parsed_precise_queries) {
    //     printf("%f, ", ele);
    // }
    // printf("\n");

    return parsed_precise_queries;
}

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

    // SPDLOG_INFO("Fetched query parms-> m_Subquantizers = {}",
    // m_Subquantizers);

    return {centroids, encrypted_parms};
}

void Client::init_client_encrypt_parms(
    const std::vector<seal::seal_byte> &serde_encrypt_parms) {

    seal::EncryptionParameters encrypt_parms;
    encrypt_parms.load(serde_encrypt_parms.data(), serde_encrypt_parms.size());
    seal::SEALContext seal_ctx(encrypt_parms);

    m_OptEncryption.emplace(encrypt_parms, seal_ctx);

    // SPDLOG_INFO("Encrypted parms: Poly modulus degree = {}",
    //             m_OptEncryption->m_EncryptedParms.poly_modulus_degree());
}

std::pair<std::vector<faiss_idx_t>, std::vector<faiss_idx_t>>
Client::sort_nearest_centroids(std::vector<float> &precise_queries,
                               std::vector<float> &centroids) const {

    std::vector<faiss_idx_t> computed_nearest_centroids_idx;
    computed_nearest_centroids_idx.reserve(m_NumQueries * m_Nlist);
    std::vector<faiss_idx_t> nprobe_nearest_centroids_idx;
    nprobe_nearest_centroids_idx.reserve(m_NumQueries * m_NProbe);

    std::vector<DistanceIndexData> nquery_centroids_distance;
    nquery_centroids_distance.reserve(m_NumQueries * m_Nlist);

    // Iterating over nqueries
    for (int i = 0; i < m_NumQueries; i++) {
        std::span<float> precise_query_view(precise_queries.data() +
                                                (i * m_PreciseVectorDimensions),
                                            m_PreciseVectorDimensions);

        // Distance wrt each centroid
        for (int j = 0; j < m_Nlist; j++) {
            std::span<float> centroid_view(centroids.data() +
                                               (j * m_PreciseVectorDimensions),
                                           m_PreciseVectorDimensions);
            float distance = 0.0;

            for (int k = 0; k < m_PreciseVectorDimensions; k++) {
                distance +=
                    std::pow(precise_query_view[k] - centroid_view[k], 2);
            }
            nquery_centroids_distance.push_back(DistanceIndexData{
                distance,
                j,
            });
        }
    }

    for (size_t i = 0; i < m_NumQueries; i++) {
        std::span<DistanceIndexData> query_centroid_view(
            nquery_centroids_distance.data() + (i * m_Nlist), m_Nlist);

        std::ranges::sort(query_centroid_view, [&](const DistanceIndexData &a,
                                                   const DistanceIndexData &b) {
            return a.distance < b.distance;
        });

        // SPDLOG_INFO("\n\n QUERY = {}", i);
        for (const DistanceIndexData &query_centroid : query_centroid_view) {
            computed_nearest_centroids_idx.push_back(query_centroid.idx);
            // SPDLOG_INFO("Centroid idx={}, distance = {}", query_centroid.idx,
            //             query_centroid.distance);
        }

        std::span<faiss_idx_t> nprobe_query_centroids(
            computed_nearest_centroids_idx.data() + i * m_Nlist, m_NProbe);
        nprobe_nearest_centroids_idx.insert(nprobe_nearest_centroids_idx.end(),
                                            nprobe_query_centroids.begin(),
                                            nprobe_query_centroids.end());
    }

    return {computed_nearest_centroids_idx, nprobe_nearest_centroids_idx};
}

std::tuple<std::vector<std::vector<std::vector<seal::seal_byte>>>,
           std::vector<std::vector<std::vector<seal::seal_byte>>>,
           std::vector<seal::seal_byte>, std::vector<seal::seal_byte>>
Client::compute_encrypted_coarse_search_parms(
    std::vector<float> &precise_queries, std::vector<float> &centroids,
    std::vector<faiss_idx_t> &nearest_centroids_idx) const {

    SPDLOG_INFO("Computing encrypted coarse search parms");

    std::vector<seal::seal_byte> serde_relin_keys(
        m_OptEncryption.value().SerdeRelinKeys.save_size());
    m_OptEncryption.value().SerdeRelinKeys.save(serde_relin_keys.data(),
                                                serde_relin_keys.size());

    std::vector<seal::seal_byte> serde_galois_keys(
        m_OptEncryption.value().SerdeGaloisKeys.save_size());
    m_OptEncryption.value().SerdeGaloisKeys.save(serde_galois_keys.data(),
                                                 serde_galois_keys.size());

    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_nqueries_residual_vectors;
    serde_nqueries_residual_vectors.reserve(m_NumQueries);
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_nqueries_residual_vectors_squared;
    serde_nqueries_residual_vectors_squared.reserve(m_NumQueries);

    if (!m_OptEncryption.has_value()) {
        SPDLOG_ERROR("Encryption uninitialised");
        throw std::runtime_error("Encryption uninitialised");
    }

    if (m_PreciseVectorDimensions >
        m_OptEncryption.value().EncryptedParms.poly_modulus_degree()) {
        SPDLOG_ERROR("Elements per vector exceeds poly modulus degree");
        throw std::runtime_error(
            "Elements per vector exceeds poly modulus degree");
    }

    for (int i = 0; i < m_NumQueries; i++) {
        std::span<float> query_vector(precise_queries.data() +
                                          i * m_PreciseVectorDimensions,
                                      m_PreciseVectorDimensions);
        std::vector<std::vector<seal::seal_byte>> serde_residual_vectors;
        serde_residual_vectors.reserve(m_NProbe);
        std::vector<std::vector<seal::seal_byte>>
            serde_residual_vectors_squared;
        serde_nqueries_residual_vectors.reserve(m_NProbe);

        SPDLOG_INFO("\n\n Query num = {}", i);
        SPDLOG_INFO("Printing Query");
        for (int temp = 0; temp < m_PreciseVectorDimensions; temp++) {
            printf("%f, ", query_vector[temp]);
        }
        printf("\n");

        for (int k = 0; k < m_NProbe; k++) {
            std::span<float> centroid_view(
                // i * m_Nlist - offset for nth query
                // k - nprobe for each query
                centroids.data() + nearest_centroids_idx[i * m_Nlist + k] *
                                       m_PreciseVectorDimensions,
                m_PreciseVectorDimensions);
            std::vector<int64_t> int_residual_query_vector(
                m_PreciseVectorDimensions, 0LL);

            SPDLOG_INFO("Query num = {}, nprobe = {}, nearest centroid = {}", i,
                        k, nearest_centroids_idx[k]);
            SPDLOG_INFO("Printing Centroid");
            for (int temp = 0; temp < m_PreciseVectorDimensions; temp++) {
                printf("%f, ", centroid_view[temp]);
            }
            printf("\n");

            std::vector<float> residual_query_vector;
            residual_query_vector.reserve(m_PreciseVectorDimensions);
            std::transform(query_vector.begin(), query_vector.end(),
                           centroid_view.begin(), residual_query_vector.begin(),
                           std::minus<float>());

            float residual_vector_squared = 0;
            for (int k = 0; k < m_PreciseVectorDimensions; k++) {
                int_residual_query_vector[k] = static_cast<int64_t>(
                    residual_query_vector[k] * BFV_SCALING_FACTOR);
                residual_vector_squared +=
                    (std::pow(residual_query_vector[k], 2));
            }
            SPDLOG_INFO("Printing computed residual, square = {}",
                        residual_vector_squared);
            for (int temp = 0; temp < m_PreciseVectorDimensions; temp++) {
                printf("%f -> %lld, ", residual_query_vector[temp],
                       int_residual_query_vector[temp]);
            }
            printf("\n\n");

            seal::Plaintext pt_residual_query_vector;
            m_OptEncryption.value().BatchEncoder.encode(
                int_residual_query_vector, pt_residual_query_vector);

            seal::Serializable<seal::Ciphertext>
                encrypted_residual_query_vector =
                    m_OptEncryption.value().Encryptor.encrypt_symmetric(
                        pt_residual_query_vector);

            std::vector<seal::seal_byte> serde_residual_ind_vector;
            serde_residual_ind_vector.resize(
                encrypted_residual_query_vector.save_size());
            encrypted_residual_query_vector.save(
                serde_residual_ind_vector.data(),
                serde_residual_ind_vector.size());

            std::vector<int64_t> encoded_residual_vector_squared(1, 0LL);
            encoded_residual_vector_squared[0] = residual_vector_squared *
                                                 BFV_SCALING_FACTOR *
                                                 BFV_SCALING_FACTOR;
            seal::Plaintext pt_residual_vector_squared;
            m_OptEncryption.value().BatchEncoder.encode(
                encoded_residual_vector_squared, pt_residual_vector_squared);
            seal::Serializable<seal::Ciphertext> encrypted_vector_ind_squared =
                m_OptEncryption.value().Encryptor.encrypt_symmetric(
                    pt_residual_vector_squared);

            std::vector<seal::seal_byte> serde_residual_ind_vector_squared;
            serde_residual_ind_vector_squared.resize(
                encrypted_vector_ind_squared.save_size());
            encrypted_vector_ind_squared.save(
                (serde_residual_ind_vector_squared.data()),
                serde_residual_ind_vector_squared.size());

            serde_residual_vectors.push_back(serde_residual_ind_vector);
            serde_residual_vectors_squared.push_back(
                serde_residual_ind_vector_squared);
        }

        serde_nqueries_residual_vectors.push_back(serde_residual_vectors);
        serde_nqueries_residual_vectors_squared.push_back(
            serde_residual_vectors_squared);
    }

    return {serde_nqueries_residual_vectors,
            serde_nqueries_residual_vectors_squared, serde_relin_keys,
            serde_galois_keys};
}

std::pair<std::vector<std::vector<std::vector<seal::seal_byte>>>,
          std::vector<std::vector<faiss_idx_t>>>
Client::get_encrypted_coarse_scores(
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        &serde_encrypted_vecs,
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        &serde_encrypted_vecs_squared,
    std::vector<faiss_idx_t> &nprobe_nearest_centroids_idx,
    std::vector<seal::seal_byte> &serde_relin_keys,
    std::vector<seal::seal_byte> &serde_galois_keys) const {

    nlohmann::json coarse_search_json;
    coarse_search_json["numQueries"] = m_NumQueries;
    coarse_search_json["residualVecs"] = serde_encrypted_vecs;
    coarse_search_json["residualVecsSquared"] = serde_encrypted_vecs_squared;
    coarse_search_json["nearestCentroids"] = nprobe_nearest_centroids_idx;
    coarse_search_json["relinKeys"] = serde_relin_keys;
    coarse_search_json["galoisKeys"] = serde_galois_keys;

    // TODO: Remove sk, used for debugging
    std::vector<seal::seal_byte> serde_sk(
        m_OptEncryption.value().SecretKey.save_size());
    m_OptEncryption.value().SecretKey.save(serde_sk.data(), serde_sk.size());
    coarse_search_json["sk"] = serde_sk;

    // SPDLOG_INFO("residualVecs Size = {}, residualVecsSquared Size = {}",
    //             coarse_search_json["residualVecs"].dump().size(),
    //             coarse_search_json["residualVecsSquared"].dump().size());
    SPDLOG_INFO("Size of the coarse search request = {}",
                coarse_search_json.dump().size());

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);

    SPDLOG_INFO("resp ={}", resp.dump());
    std::vector<std::vector<std::vector<seal::seal_byte>>>
        encrypted_coarse_scores;
    std::vector<std::vector<faiss_idx_t>> coarse_vector_labels;

    // auto encrypted_coarse_scores =
    //     resp.at("encryptedCoarseDistances")
    //         .get<std::vector<std::vector<std::vector<seal::seal_byte>>>>();
    //
    // auto coarse_vector_labels =
    //     resp.at("coarseVectorLabels")
    //         .get<std::vector<std::vector<faiss_idx_t>>>();

    return {encrypted_coarse_scores, coarse_vector_labels};
}

std::vector<std::vector<float>> Client::deserialise_decrypt_coarse_distances(
    const std::vector<std::vector<std::vector<seal::seal_byte>>>
        &serde_encrypted_coarse_distances) {

    std::vector<std::vector<float>> nquery_coarse_distances;
    nquery_coarse_distances.reserve(serde_encrypted_coarse_distances.size());
    seal::SEALContext seal_ctx(m_OptEncryption.value().EncryptedParms);

    SPDLOG_INFO("Deserialising decrypted coarse distances");

    for (int i = 0; i < serde_encrypted_coarse_distances.size(); i++) {
        std::vector<float> nprobe_coarse_distances;
        nprobe_coarse_distances.reserve(
            serde_encrypted_coarse_distances[i].size());

        for (int j = 0; j < serde_encrypted_coarse_distances[i].size(); j++) {

            seal::Ciphertext encrypted_coarse_distance;
            seal::Plaintext decrypted_coarse_distance;
            std::vector<int64_t> decoded_coarse_distances;
            float coarse_distances;

            encrypted_coarse_distance.load(
                seal_ctx, serde_encrypted_coarse_distances[i][j].data(),
                serde_encrypted_coarse_distances[i][j].size());
            m_OptEncryption.value().Decryptor.decrypt(
                encrypted_coarse_distance, decrypted_coarse_distance);
            m_OptEncryption.value().BatchEncoder.decode(
                decrypted_coarse_distance, decoded_coarse_distances);
            coarse_distances = static_cast<float>(decoded_coarse_distances[0]) /
                               (BFV_SCALING_FACTOR * BFV_SCALING_FACTOR);
            nprobe_coarse_distances.push_back(coarse_distances);
        }

        nquery_coarse_distances.push_back(nprobe_coarse_distances);
    }

    SPDLOG_INFO("Printing deserialised decrypted coarse distances");
    for (int i = 0; i < nquery_coarse_distances.size(); i++) {
        for (int j = 0; j < nquery_coarse_distances[i].size(); j++) {
            printf("%f, ", nquery_coarse_distances[i][j]);
        }
        printf("\n");
    }

    return nquery_coarse_distances;
}

std::vector<std::vector<faiss_idx_t>>
Client::compute_nearest_coarse_vectors_idx(
    const std::vector<std::vector<float>> &decrypted_coarse_distance_scores,
    const std::vector<std::vector<faiss_idx_t>> &coarse_vector_labels,
    const size_t num_queries, const size_t coarse_probe) const {
    std::vector<std::vector<DistanceIndexData>> nquery_coarse_vector_distances;
    nquery_coarse_vector_distances.reserve(num_queries);

    std::vector<std::vector<faiss_idx_t>> nquery_coarse_vector;
    nquery_coarse_vector.reserve(num_queries);

    for (int i = 0; i < num_queries; i++) {
        if (decrypted_coarse_distance_scores[i].size() < coarse_probe) {
            SPDLOG_ERROR("Number of computed coarse scores is lesser than "
                         "coarse_probe");
            throw std::runtime_error("Number of computed coarse scores is "
                                     "lesser than coarse_probe");
        }

        std::vector<DistanceIndexData> decrypted_per_query_vector_distances;
        decrypted_per_query_vector_distances.reserve(
            decrypted_coarse_distance_scores[i].size());

        for (int j = 0; j < decrypted_coarse_distance_scores[i].size(); j++) {
            decrypted_per_query_vector_distances.push_back(DistanceIndexData{
                decrypted_coarse_distance_scores[i][j],
                coarse_vector_labels[i][j],
            });
        }

        nquery_coarse_vector_distances.push_back(
            decrypted_per_query_vector_distances);
    }

    for (int k = 0; k < num_queries; k++) {

        std::vector<faiss_idx_t> coarse_probe_query_vector;
        coarse_probe_query_vector.reserve(coarse_probe);
        std::ranges::sort(
            nquery_coarse_vector_distances[k],
            [&](const DistanceIndexData &a, const DistanceIndexData &b) {
                return a.distance < b.distance;
            });

        std::span coarse_probe_view(nquery_coarse_vector_distances[k].begin(),
                                    coarse_probe);
        std::transform(coarse_probe_view.begin(), coarse_probe_view.end(),
                       coarse_probe_query_vector.begin(),
                       [](const DistanceIndexData &dt) { return dt.idx; });
        nquery_coarse_vector.push_back(coarse_probe_query_vector);
    }

    SPDLOG_INFO("Printing nearest coarse_probe vectors");
    SPDLOG_INFO("nquery_coarse_vector.size() = {}",
                nquery_coarse_vector.size());
    for (int i = 0; i < nquery_coarse_vector.size(); i++) {
        SPDLOG_INFO("nquery_coarse_vector[{}].size() = {}", i,
                    nquery_coarse_vector[i].size());
        for (int j = 0; j < nquery_coarse_vector[i].size(); j++) {
            printf("%lld, ", nquery_coarse_vector[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return nquery_coarse_vector;
}

void get_precise_scores(
    const std::array<std::vector<DistanceIndexData>, NQUERY>
        &sorted_coarse_vectors,
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores) {

    std::array<std::array<faiss_idx_t, COARSE_PROBE>, NQUERY>
        nearest_coarse_vectors_id;

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < COARSE_PROBE; j++) {
            nearest_coarse_vectors_id[i][j] = sorted_coarse_vectors[i][j].idx;
        }
    }

    nlohmann::json coarse_search_json;
    coarse_search_json["preciseQuery"] = precise_query;
    coarse_search_json["nearestCoarseVectorIndexes"] =
        nearest_coarse_vectors_id;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "precisesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);

    precise_scores =
        resp.at("preciseDistanceScores")
            .get<std::array<std::array<float, COARSE_PROBE>, NQUERY>>();
}

void compute_nearest_precise_vectors(
    const std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores,
    const std::array<std::vector<DistanceIndexData>, NQUERY>
        &sorted_coarse_vectors,
    std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
        &nearest_precise_vectors) {
    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < COARSE_PROBE; j++) {
            nearest_precise_vectors[i][j] = DistanceIndexData{
                precise_scores[i][j], sorted_coarse_vectors[i][j].idx};
        }
    }

    for (auto &precise_score_query : nearest_precise_vectors) {
        std::ranges::sort(precise_score_query, [&](const DistanceIndexData &a,
                                                   const DistanceIndexData &b) {
            return a.distance < b.distance;
        });
    }
}

void get_precise_vectors_pir(
    const std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
        &nearest_precise_vectors,
    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY> &query_results,
    std::array<std::array<faiss_idx_t, K>, NQUERY> &query_results_idx) {

    if constexpr (K > COARSE_PROBE) {
        SPDLOG_ERROR("K greater than COARSE_PROBE");
        throw std::runtime_error("K greater than COARSE_PROBE");
    }

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < K; j++) {
            query_results_idx[i][j] = nearest_precise_vectors[i][j].idx;
        }
    }

    nlohmann::json precise_vector_pir_json;
    precise_vector_pir_json["nearestPreciseVectorIndexes"] = query_results_idx;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "precise-vector-pir"),
                                cpr::Body(precise_vector_pir_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);

    query_results =
        resp.at("queryResults")
            .get<std::array<
                std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                NQUERY>>();
}

void benchmark_results(const std::array<std::array<faiss_idx_t, K>, NQUERY>
                           &observed_query_results_idx) {

    SPDLOG_INFO("BENCHMARK RESULTS");

    size_t gt_nn_per_query;
    size_t gt_nq;
    std::vector<int> ground_truth;
    vecs_read(GROUNDTRUTH_DATASET_PATH, gt_nn_per_query, gt_nq, ground_truth);

    // MRR considers only the position of the first result
    // MRR@10 - Results outside top 10 are irrelevant
    float mrr_1 = 0, mrr_10 = 0, mrr_100 = 0;

    // Recall considers ratio of true results to ground truth
    int nq_recall_1 = 0, nq_recall_10 = 0, nq_recall_100 = 0;

    if (K > gt_nn_per_query) {
        SPDLOG_ERROR("K greater than nearest neigbours per query in ground "
                     "truth dataset");
        throw std::runtime_error(
            "K greater than nearest neigbours per query in ground "
            "truth dataset");
    }
    for (int i = 0; i < NQUERY; i++) {
        // printf("\n\n");
        // SPDLOG_INFO("QUERY BENCHMARKS Q = {}", i + 1);
        // SPDLOG_INFO("Ground truth nearest neighbours for Q = {}", i + 1);
        int recall_1 = 0, recall_10 = 0, recall_100 = 0;
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < K; k++) {
                if (ground_truth[i * gt_nn_per_query + j] ==
                    observed_query_results_idx[i][k]) {
                    if (k < 1)
                        recall_1++;
                    if (k < 10)
                        recall_10++;
                    if (k < 100)
                        recall_100++;

                    // Considering only 1st ground truth for MRR
                    if (j == 0) {
                        if (k < 1)
                            mrr_1 += 1.0f / static_cast<float>(k + 1);
                        if (k < 10)
                            mrr_10 += 1.0f / static_cast<float>(k + 1);
                        if (k < 100)
                            mrr_100 += 1.0f / static_cast<float>(k + 1);

                        // SPDLOG_INFO("Updated mrr = {}, {}, {}", mrr_1,
                        // mrr_10,
                        //             mrr_100);
                    }
                    break;
                }
            }
            // printf("%d, ", ground_truth[i * gt_nn_per_query + j]);
        }
        // printf("\n");

        // SPDLOG_INFO("Query Results:");
        // for (int j = 0; j < K; j++) {
        //     printf("%lld, ", observed_query_results_idx[i][j]);
        // }
        // printf("\n");

        // SPDLOG_INFO("Recall@1 = {}, Recall@10 = {}, Recall@100 = {}",
        //             static_cast<float>(recall_1) / 1,
        //             static_cast<float>(recall_10) / 10,
        //             static_cast<float>(recall_100) / 100);
        nq_recall_1 += recall_1;
        nq_recall_10 += recall_10;
        nq_recall_100 += recall_100;
    }

    printf("\n\n");
    SPDLOG_INFO("Total Query Benchmark Results");
    SPDLOG_INFO("Parameters: NPROBE = {}, COARSE_PROBE = {}, K = {}", NPROBE,
                COARSE_PROBE, K);
    SPDLOG_INFO("Parameters: NQUERY = {}, NLIST = {}", NQUERY, NLIST);
    SPDLOG_INFO("Parameters: SUB_QUANTIZERS = {}, SUB_VECTOR_SIZE = {}",
                SUB_QUANTIZERS, SUB_QUANTIZER_SIZE);
    SPDLOG_INFO("Recall@1 = {}, Recall@10 = {}, Recall@100 = {}",
                static_cast<float>(nq_recall_1) / (1 * NQUERY),
                static_cast<float>(nq_recall_10) / (10 * NQUERY),
                static_cast<float>(nq_recall_100) / (100 * NQUERY));
    SPDLOG_INFO("MRR@1 = {}, MRR@10 = {}, MRR@100 = {}", (float)mrr_1 / NQUERY,
                (float)mrr_10 / NQUERY, (float)mrr_100 / NQUERY);

    printf("\n\n");
    for (int i = 0; i < 100; i++) {
        printf("-");
    }
    printf("\n");
}
