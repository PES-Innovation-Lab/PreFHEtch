#include <cstdio>
#include <vector>

#include "faiss/MetricType.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_server_utils.h"
#include "server_lib.h"

// Include controllers headers to register with server
#include "controllers/Query.h"

char const *SERVER_ADDRESS = "0.0.0.0";
constexpr int SERVER_PORT = 8080;

char const *TRAIN_DATASET_PATH = "../sift/siftsmall/siftsmall_learn.fvecs";
char const *BASE_DATASET_PATH = "../sift/siftsmall/siftsmall_base.fvecs";

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

// Path - build/_.faiss
std::string INDEX_FILE;

Server::Server() : m_EncryptionParms(seal::scheme_type::bfv) {
    std::ostringstream oss;
    oss << "IVF" << m_Nlist << "_PQ" << m_SubQuantizers << "_SUB_QUANTIZER_SIZE"
        << m_SubQuantizerSize << ".faiss";
    INDEX_FILE = oss.str();

    m_PolyModulusDegree = 4096;
    // Setting to same size as float 32 to prevent overflow
    m_PlaintextModulusSize = 32;

    m_EncryptionParms.set_poly_modulus_degree(m_PolyModulusDegree);
    m_EncryptionParms.set_coeff_modulus(
        seal::CoeffModulus::BFVDefault(m_PolyModulusDegree));
    m_EncryptionParms.set_plain_modulus(seal::PlainModulus::Batching(
        m_PolyModulusDegree, m_PlaintextModulusSize));
}

void Server::run_webserver() {
    drogon::app().addListener(SERVER_ADDRESS, SERVER_PORT);

    SPDLOG_INFO("Server listening on {}:{}", SERVER_ADDRESS, SERVER_PORT);
    drogon::app().run();
}

void Server::init_index() {
    // Training the index
    if (!(std::filesystem::exists(INDEX_FILE))) {
        SPDLOG_INFO("Loading train set");

        size_t parsed_training_count;
        std::vector<float> parsed_train_set;
        vecs_read<float>(TRAIN_DATASET_PATH, m_PreciseVectorDimensions,
                         parsed_training_count, parsed_train_set);

        m_Quantizer = faiss::IndexFlatL2(m_PreciseVectorDimensions);
        m_Index = std::make_unique<faiss::IndexIVFPQ>(
            &m_Quantizer, m_PreciseVectorDimensions, m_Nlist, m_SubQuantizers,
            m_SubQuantizerSize);

        SPDLOG_INFO("Training on {} vectors", parsed_training_count);
        m_Index->train(parsed_training_count, parsed_train_set.data());

        SPDLOG_INFO("Loading database");

        size_t nb, d2;
        vecs_read<float>(BASE_DATASET_PATH, d2, nb, m_DatasetBase);
        assert(m_PreciseVectorDimensions == d2 ||
               !"dataset does not have same dimension as train set");

        SPDLOG_INFO("Indexing database, Records = {}, Dimensions = {}", nb,
                    m_PreciseVectorDimensions);
        m_Index->add(nb, m_DatasetBase.data());

        faiss::write_index(m_Index.get(), INDEX_FILE.c_str());
        SPDLOG_INFO("Cached dataset to index file - {}", INDEX_FILE);

    } else {
        SPDLOG_INFO("Reading cached data from index file - {}", INDEX_FILE);

        vecs_read<float>(BASE_DATASET_PATH, m_PreciseVectorDimensions, m_NBase,
                         m_DatasetBase);

        faiss::Index *loaded_ptr = faiss::read_index(INDEX_FILE.c_str());
        auto *loaded_ivfpq = dynamic_cast<faiss::IndexIVFPQ *>(loaded_ptr);
        if (!loaded_ivfpq) {
            throw std::runtime_error("Loaded index is not of type IndexIVFPQ");
        }

        m_Index.reset(loaded_ivfpq);
    }
}

std::vector<float> Server::retrieve_centroids() const {
    std::vector<float> centroids;
    centroids.resize(m_Nlist * m_PreciseVectorDimensions);

    for (int i = 0; i < m_Nlist; i++) {
        m_Index->quantizer->reconstruct(i, centroids.data() +
                                               (i * m_PreciseVectorDimensions));
    }

    return centroids;
}

std::vector<seal::seal_byte> Server::serialise_parms() const {
    std::vector<seal::seal_byte> serde_parms;
    return serde_parms;
}

void Server::coarseSearch(
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    const std::array<std::array<faiss::idx_t, NPROBE>, NQUERY>
        &nearest_centroid_idx,
    std::vector<float> &coarse_distance_scores,
    std::vector<faiss::idx_t> &coarse_distance_indexes,
    std::array<size_t, NQUERY> &list_sizes_per_query) const {
    // Reset nprobe, previously set by auto-tuning
    m_Index->nprobe = NPROBE;

    coarse_distance_scores.resize(NBASE * NQUERY);
    coarse_distance_indexes.resize(NBASE * NQUERY);

    m_Index->search_encrypted(
        NQUERY, precise_query.data()->data(),
        const_cast<faiss::idx_t *>(nearest_centroid_idx.data()->data()),
        coarse_distance_scores.data(), coarse_distance_indexes.data(),
        list_sizes_per_query.data());

    size_t coarse_vectors_count = std::accumulate(
        list_sizes_per_query.begin(), list_sizes_per_query.end(), 0);
    coarse_distance_scores.resize(coarse_vectors_count);
    coarse_distance_indexes.resize(coarse_vectors_count);

    // SPDLOG_INFO("Coarse search complete");
}

void Server::preciseSearch(
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    const std::array<std::array<faiss::idx_t, COARSE_PROBE>, NQUERY>
        &nearest_coarse_vector_idx,
    std::array<std::array<float, COARSE_PROBE>, NQUERY>
        &precise_distance_scores) const {
    // SPDLOG_INFO("Starting precise search on the server");

    const float *dataset_base_ptr = m_DatasetBase.data();

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < COARSE_PROBE; j++) {
            float dist = 0.0;
            const float *precise_vec_idx =
                dataset_base_ptr +
                (nearest_coarse_vector_idx[i][j] * PRECISE_VECTOR_DIMENSIONS);

            for (int k = 0; k < PRECISE_VECTOR_DIMENSIONS; k++) {
                dist += std::pow((precise_vec_idx[k] - precise_query[i][k]), 2);
            }

            precise_distance_scores[i][j] = dist;
        }
    }

    // SPDLOG_INFO("Precise search completed");
}

void Server::preciseVectorPIR(
    const std::array<std::array<faiss_idx_t, K>, NQUERY>
        &k_nearest_precise_vectors_idx,
    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY> &query_results) {
    // SPDLOG_INFO("Starting precise vector PIR on the server");

    float *dataset_base_ptr = m_DatasetBase.data();

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < K; j++) {
            float *precise_vec_idx =
                dataset_base_ptr + (k_nearest_precise_vectors_idx[i][j] *
                                    PRECISE_VECTOR_DIMENSIONS);

            std::span<float> prec_vec(precise_vec_idx,
                                      PRECISE_VECTOR_DIMENSIONS *
                                          sizeof(precise_vec_idx));
            for (int k = 0; k < PRECISE_VECTOR_DIMENSIONS; k++) {
                // SPDLOG_INFO("Vector {} Dimension k[{}] = {}", j, k,
                // prec_vec[k]);
                query_results[i][j][k] = prec_vec[k];
            }
        }
    }

    // SPDLOG_INFO("Precise vector PIR completed");
}
