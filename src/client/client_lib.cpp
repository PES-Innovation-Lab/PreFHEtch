#include <algorithm>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

#include "../../build/_deps/prefhetch-faiss-src/faiss/utils/bf16.h"
#include "client_server_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

Encryption::Encryption(seal::EncryptionParameters encrypt_parms,
                       const seal::SEALContext &seal_ctx)
    : EncryptedParms(std::move(encrypt_parms)), KeyGen(seal_ctx),
      SecretKey(KeyGen.secret_key()), SerRelinKeys(KeyGen.create_relin_keys()),
      Encryptor(seal_ctx, SecretKey), Decryptor(seal_ctx, SecretKey),
      BatchEncoder(seal_ctx) {}

Client::Client(size_t num_queries) { m_NumQueries = num_queries; }

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

    // SPDLOG_INFO("Fetched query parms-> m_Subquantizers = {}",
    // m_Subquantizers);

    return {centroids, encrypted_parms};
}

void Client::init_client_encrypt_parms(
    const std::vector<seal::seal_byte> &serde_encrypt_parms) {

    seal::EncryptionParameters encrypt_parms;
    encrypt_parms.load(
        reinterpret_cast<const seal::seal_byte *>(serde_encrypt_parms.data()),
        serde_encrypt_parms.size());
    seal::SEALContext seal_ctx(encrypt_parms);

    m_OptEncryption.emplace(encrypt_parms, seal_ctx);

    // SPDLOG_INFO("Encrypted parms: Poly modulus degree = {}",
    //             m_OptEncryption->m_EncryptedParms.poly_modulus_degree());
}

std::pair<std::vector<std::vector<seal::seal_byte>>,
          std::vector<std::vector<seal::seal_byte>>>
Client::compute_encrypted_subvector_components(
    std::vector<float> &precise_queries) const {
    std::vector<std::vector<seal::seal_byte>> serde_subvectors;
    std::vector<std::vector<seal::seal_byte>> serde_subvectors_squared;
    serde_subvectors.reserve(m_NumQueries * m_Subquantizers);
    serde_subvectors_squared.reserve(m_NumQueries * m_Subquantizers);

    if (!m_OptEncryption.has_value()) {
        SPDLOG_ERROR("Encryption uninitialised");
        throw std::runtime_error("Encryption uninitialised");
    }

    size_t elements_per_subvector = m_PreciseVectorDimensions / m_Subquantizers;
    if (elements_per_subvector >
        m_OptEncryption.value().EncryptedParms.poly_modulus_degree()) {
        SPDLOG_ERROR("Elements per subvector exceeds poly modulus degree");
        throw std::runtime_error(
            "Elements per subvector exceeds poly modulus degree");
    }

    std::vector<uint64_t> int_query_subvector(elements_per_subvector, 0ULL);

    for (int i = 0; i < m_Subquantizers * m_NumQueries; i++) {
        std::span<float> query_subvector_view(precise_queries.data() +
                                                  i * elements_per_subvector,
                                              elements_per_subvector);

        float subvector_len_squared = 0;
        for (int j = 0; j < elements_per_subvector; j++) {
            int_query_subvector[j] = static_cast<uint64_t>(
                query_subvector_view[j] * BFV_SCALING_FACTOR);
            subvector_len_squared += (std::pow(query_subvector_view[j], 2));
        }
        subvector_len_squared *= BFV_SCALING_FACTOR;

        seal::Plaintext pt_query_subvector;
        m_OptEncryption.value().BatchEncoder.encode(int_query_subvector,
                                                    pt_query_subvector);
        int_query_subvector.clear();

        seal::Serializable<seal::Ciphertext> encrypted_query_subvector =
            m_OptEncryption.value().Encryptor.encrypt_symmetric(
                pt_query_subvector);

        std::vector<seal::seal_byte> serde_query_subvector;
        serde_query_subvector.resize(encrypted_query_subvector.save_size());
        encrypted_query_subvector.save(serde_query_subvector.data(),
                                       serde_query_subvector.size());

        uint64_t u64_subvector_len =
            static_cast<uint64_t>(subvector_len_squared);
        seal::Plaintext pt_subvector_len_squared(
            seal::util::uint_to_hex_string(&u64_subvector_len, std::size_t(1)));
        seal::Serializable<seal::Ciphertext> encrypted_subvector_len_squared =
            m_OptEncryption.value().Encryptor.encrypt_symmetric(
                pt_subvector_len_squared);

        std::vector<seal::seal_byte> serde_subvector_len_squared;
        serde_subvector_len_squared.resize(
            encrypted_subvector_len_squared.save_size());
        encrypted_subvector_len_squared.save(
            (serde_subvector_len_squared.data()),
            serde_subvector_len_squared.size());

        serde_subvectors.push_back(serde_query_subvector);
        serde_subvectors_squared.push_back(serde_subvector_len_squared);
    }

    return {serde_subvectors, serde_subvectors_squared};
}

std::vector<faiss_idx_t>
Client::sort_nearest_centroids(std::vector<float> &precise_queries,
                               std::vector<float> &centroids) const {
    std::vector<faiss_idx_t> computed_nearest_centroids_idx;
    computed_nearest_centroids_idx.reserve(m_Nlist * m_NumQueries);

    std::vector<DistanceIndexData> nquery_centroids_distance;
    nquery_centroids_distance.reserve(m_Nlist * m_NumQueries);

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
    }

    return computed_nearest_centroids_idx;
}

void Client::get_encrypted_coarse_scores(
    std::vector<std::vector<seal::seal_byte>> &encrypted_subvectors,
    std::vector<std::vector<seal::seal_byte>> &encrypted_subvectors_square,
    std::vector<float> &coarse_scores,
    std::vector<faiss_idx_t> &coarse_vectors_idx,
    std::vector<size_t> &list_sizes_per_query_coarse) {

    std::vector<seal::seal_byte> sk(
        m_OptEncryption.value().SecretKey.save_size());

    m_OptEncryption.value().SecretKey.save(sk.data(), sk.size());

    nlohmann::json coarse_search_json;
    coarse_search_json["numQueries"] = m_NumQueries;
    coarse_search_json["subvectors"] = encrypted_subvectors;
    coarse_search_json["subvectorsSquared"] = encrypted_subvectors_square;
    coarse_search_json["secretKey"] = sk;

    SPDLOG_INFO("Size of the coarse search request = {}",
                coarse_search_json.dump().size());

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}", resp.dump());
    //
    // coarse_scores =
    // resp.at("coarseDistanceScores").get<std::vector<float>>();
    // coarse_vectors_idx =
    //     resp.at("coarseVectorIndexes").get<std::vector<faiss_idx_t>>();
    // list_sizes_per_query =
    //     resp.at("listSizesPerQuery").get<std::array<size_t, NQUERY>>();
}

void compute_nearest_coarse_vectors(
    const std::vector<float> &coarse_distance_scores,
    const std::vector<faiss_idx_t> &coarse_vector_indexes,
    const std::array<size_t, NQUERY> &list_sizes_per_query,
    std::array<std::vector<DistanceIndexData>, NQUERY>
        &nearest_coarse_vectors) {

    size_t current_vector_index = 0;

    for (int i = 0; i < NQUERY; i++) {
        if (list_sizes_per_query[i] < COARSE_PROBE) {
            SPDLOG_ERROR("Number of computed coarse scores is lesser than "
                         "COARSE_PROBE");
            throw std::runtime_error("Number of computed coarse scores is "
                                     "lesser than COARSE_PROBE");
        }
        nearest_coarse_vectors[i].reserve(list_sizes_per_query[i]);

        for (int j = 0; j < list_sizes_per_query[i]; j++) {
            const size_t idx = current_vector_index + j;
            nearest_coarse_vectors[i].push_back(DistanceIndexData{
                coarse_distance_scores[idx],
                coarse_vector_indexes[idx],
            });
        }
        current_vector_index += list_sizes_per_query[i];
    }

    for (std::vector<DistanceIndexData> &query : nearest_coarse_vectors) {
        std::ranges::sort(
            query, [&](const DistanceIndexData &a, const DistanceIndexData &b) {
                return a.distance < b.distance;
            });
    }
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
