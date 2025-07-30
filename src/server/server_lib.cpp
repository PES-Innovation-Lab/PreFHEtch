#include <cstdint>
#include <cstdio>
#include <vector>

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
#include "seal/ciphertext.h"
#include "seal/galoiskeys.h"
#include "seal/plaintext.h"
#include "seal/relinkeys.h"
#include "seal/util/defines.h"

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
    oss << "IVF" << Nlist << "_PQ" << SubQuantizers << "_SUB_QUANTIZER_SIZE"
        << SubQuantizerSize << ".faiss";
    INDEX_FILE = oss.str();

    m_PolyModulusDegree = 8192;
    m_PlaintextModulusSize = 48;

    m_EncryptionParms.set_poly_modulus_degree(m_PolyModulusDegree);
    m_EncryptionParms.set_coeff_modulus(
        seal::CoeffModulus::BFVDefault(m_PolyModulusDegree));
    m_EncryptionParms.set_plain_modulus(seal::PlainModulus::Batching(
        m_PolyModulusDegree, m_PlaintextModulusSize));

    seal::SEALContext seal_ctx(m_EncryptionParms);
    SPDLOG_INFO("Encryption params errors = {}",
                seal_ctx.parameter_error_message());
}

void Server::run_webserver() {
    // TODO: Look into this
    drogon::app().setClientMaxBodySize(500 * 1024 * 1024);
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
            &m_Quantizer, m_PreciseVectorDimensions, Nlist, SubQuantizers,
            SubQuantizerSize);

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
    centroids.resize(Nlist * m_PreciseVectorDimensions);

    for (int i = 0; i < Nlist; i++) {
        m_Index->quantizer->reconstruct(i, centroids.data() +
                                               (i * m_PreciseVectorDimensions));
    }

    return centroids;
}

std::vector<seal::seal_byte> Server::serialise_parms() const {
    std::vector<seal::seal_byte> serde_parms;
    serde_parms.resize(static_cast<size_t>(m_EncryptionParms.save_size()));
    m_EncryptionParms.save(
        reinterpret_cast<seal::seal_byte *>(serde_parms.data()),
        serde_parms.size());

    return serde_parms;
}

std::tuple<std::vector<std::vector<seal::Ciphertext>>,
           std::vector<std::vector<seal::Ciphertext>>, seal::RelinKeys,
           seal::GaloisKeys>
Server::deserialise_coarse_search_parms(
    const std::vector<std::vector<std::vector<seal::seal_byte>>>
        &serde_encrypted_residual_vectors,
    const std::vector<std::vector<std::vector<seal::seal_byte>>>
        &serde_encrypted_residual_vectors_squared,
    const std::vector<seal::seal_byte> &serde_relin_keys,
    const std::vector<seal::seal_byte> &serde_galois_keys,
    const std::vector<seal::seal_byte> &enc_sk) const {

    seal::SEALContext seal_ctx(m_EncryptionParms);
    seal::BatchEncoder batch_encoder(seal_ctx);

    seal::RelinKeys relin_keys;
    relin_keys.load(seal_ctx, serde_relin_keys.data(), serde_relin_keys.size());

    seal::GaloisKeys galois_keys;
    galois_keys.load(seal_ctx, serde_galois_keys.data(),
                     serde_galois_keys.size());

    SPDLOG_INFO("\n\nDecrypting coarse search parms on server temporarily\n\n");
    seal::SecretKey sk;
    sk.load(seal_ctx, enc_sk.data(), enc_sk.size());
    seal::Decryptor decryptor(seal_ctx, sk);

    std::vector<std::vector<seal::Ciphertext>>
        encrypted_nquery_residual_vectors;
    std::vector<std::vector<seal::Ciphertext>>
        encrypted_nquery_residual_vectors_squared;
    encrypted_nquery_residual_vectors.reserve(
        serde_encrypted_residual_vectors.size());
    encrypted_nquery_residual_vectors_squared.reserve(
        serde_encrypted_residual_vectors_squared.size());

    SPDLOG_INFO("Deserialising encrypted coarse search parms");
    for (int i = 0; i < serde_encrypted_residual_vectors.size(); i++) {

        std::vector<seal::Ciphertext> encrypted_query_residual_vectors;
        std::vector<seal::Ciphertext> encrypted_query_residual_vectors_squared;
        encrypted_query_residual_vectors.reserve(
            serde_encrypted_residual_vectors[i].size());
        encrypted_query_residual_vectors_squared.reserve(
            serde_encrypted_residual_vectors_squared[i].size());

        for (int j = 0; j < serde_encrypted_residual_vectors[i].size(); j++) {

            seal::Ciphertext encrypted_residual_vector;
            seal::Ciphertext encrypted_residual_vector_sq;

            encrypted_residual_vector.load(
                seal_ctx, serde_encrypted_residual_vectors[i][j].data(),
                serde_encrypted_residual_vectors[i][j].size());

            encrypted_residual_vector_sq.load(
                seal_ctx, serde_encrypted_residual_vectors_squared[i][j].data(),
                serde_encrypted_residual_vectors_squared[i][j].size());

            encrypted_query_residual_vectors.push_back(
                encrypted_residual_vector);
            encrypted_query_residual_vectors_squared.push_back(
                encrypted_residual_vector_sq);

            seal::Plaintext decrypted_residual_vector;
            seal::Plaintext decrypted_residual_vector_sq;
            std::vector<int64_t> decoded_residual_vector;
            std::vector<int64_t> decoded_residual_vector_squared;

            decryptor.decrypt(encrypted_residual_vector,
                              decrypted_residual_vector);
            batch_encoder.decode(decrypted_residual_vector,
                                 decoded_residual_vector);
            std::for_each(decoded_residual_vector.begin(),
                          decoded_residual_vector.end(),
                          [](int64_t &n) { n /= BFV_SCALING_FACTOR; });

            decryptor.decrypt(encrypted_residual_vector_sq,
                              decrypted_residual_vector_sq);
            std::vector<int64_t> encoded_u64_residual_vector_squared(1, 0LL);
            batch_encoder.decode(decrypted_residual_vector_sq,
                                 encoded_u64_residual_vector_squared);

            float decrypted_residual_vector_squared =
                encoded_u64_residual_vector_squared[0];
            decrypted_residual_vector_squared /=
                (BFV_SCALING_FACTOR * BFV_SCALING_FACTOR);

            // SPDLOG_INFO("Nquery = {}, Nprobe = {}, vec size = {}, squared val
            // "
            //             "= {}, printing "
            //             "residual vector",
            //             i, j, decoded_residual_vector.size(),
            //             decrypted_residual_vector_squared);
            // for (int k = 0; k < m_PreciseVectorDimensions; k++) {
            //     printf("%lld, ", decoded_residual_vector[k]);
            // }
            // printf("\n");
        }

        encrypted_nquery_residual_vectors.push_back(
            encrypted_query_residual_vectors);
        encrypted_nquery_residual_vectors_squared.push_back(
            encrypted_query_residual_vectors_squared);
    }

    return {encrypted_nquery_residual_vectors,
            encrypted_nquery_residual_vectors_squared, relin_keys, galois_keys};
}

std::vector<std::vector<std::vector<seal::seal_byte>>>
Server::serialise_encrypted_coarse_distances(
    const std::vector<std::vector<seal::Ciphertext>>
        &encrypted_coarse_distances) const {

    std::vector<std::vector<std::vector<seal::seal_byte>>>
        nquery_encrypted_coarse_distances;
    nquery_encrypted_coarse_distances.reserve(
        encrypted_coarse_distances.size());

    for (int i = 0; i < encrypted_coarse_distances.size(); i++) {
        std::vector<std::vector<seal::seal_byte>>
            nprobe_encrypted_coarse_distances;
        nprobe_encrypted_coarse_distances.reserve(
            encrypted_coarse_distances[i].size());

        for (int j = 0; j < encrypted_coarse_distances[i].size(); j++) {
            std::vector<seal::seal_byte> serde_encrypted_coarse_distances(
                encrypted_coarse_distances[i][j].save_size());

            encrypted_coarse_distances[i][j].save(
                serde_encrypted_coarse_distances.data(),
                serde_encrypted_coarse_distances.size());

            nprobe_encrypted_coarse_distances.push_back(
                serde_encrypted_coarse_distances);
        }

        nquery_encrypted_coarse_distances.push_back(
            nprobe_encrypted_coarse_distances);
    }

    return nquery_encrypted_coarse_distances;
}

std::pair<std::vector<std::vector<seal::Ciphertext>>,
          std::vector<std::vector<faiss::idx_t>>>
Server::coarseSearch(
    std::vector<faiss::idx_t> nprobe_centroids,
    std::vector<std::vector<seal::Ciphertext>> &encrypted_residual_queries,
    std::vector<std::vector<seal::Ciphertext>>
        &encrypted_residual_queries_squared,
    size_t num_queries, size_t nprobe, seal::RelinKeys relin_keys,
    seal::GaloisKeys galois_keys) const {

    m_Index->nprobe = nprobe;
    if (m_Index->nprobe != nprobe) {
        SPDLOG_ERROR("m_Index->nprobe != nprobe");
        throw std::runtime_error("m_Index->nprobe != nprobe");
    }

    SPDLOG_INFO("Coarse search started");

    seal::SEALContext seal_ctx(m_EncryptionParms);
    seal::BatchEncoder batch_encoder(seal_ctx);
    seal::Evaluator evaluator(seal_ctx);

    std::vector<std::vector<seal::Ciphertext>> encrypted_coarse_distances(
        num_queries * nprobe);
    std::vector<std::vector<faiss::idx_t>> coarse_distance_labels(num_queries *
                                                                  nprobe);

    m_Index->search_encrypted(
        batch_encoder, evaluator, relin_keys, galois_keys, BFV_SCALING_FACTOR,
        num_queries, encrypted_residual_queries,
        encrypted_residual_queries_squared, nprobe_centroids.data(),
        encrypted_coarse_distances, coarse_distance_labels);

    SPDLOG_INFO("Coarse search complete");

    return {encrypted_coarse_distances, coarse_distance_labels};
}

std::tuple<std::vector<seal::Ciphertext>, seal::RelinKeys, seal::GaloisKeys>
Server::deserialise_precise_search_params(
    const std::vector<std::vector<seal::seal_byte>>
        &serde_encrypted_precise_queries,
    const std::vector<seal::seal_byte> &serde_relin_keys,
    const std::vector<seal::seal_byte> &serde_galois_keys) const {

    seal::SEALContext seal_ctx(m_EncryptionParms);
    seal::BatchEncoder batch_encoder(seal_ctx);

    seal::RelinKeys relin_keys;
    relin_keys.load(seal_ctx, serde_relin_keys.data(), serde_relin_keys.size());

    seal::GaloisKeys galois_keys;
    galois_keys.load(seal_ctx, serde_galois_keys.data(),
                     serde_galois_keys.size());

    std::vector<seal::Ciphertext> encrypted_precise_queries;
    encrypted_precise_queries.reserve(serde_encrypted_precise_queries.size());

    SPDLOG_INFO("Deserializing encrypted precise search params");
    for (int i = 0; i < serde_encrypted_precise_queries.size(); i++) {
        seal::Ciphertext encrypted_precise_query;

        encrypted_precise_query.load(seal_ctx,
                                     serde_encrypted_precise_queries[i].data(),
                                     serde_encrypted_precise_queries[i].size());

        encrypted_precise_queries.push_back(encrypted_precise_query);
    }

    return {encrypted_precise_queries, relin_keys, galois_keys};
}

std::vector<std::vector<seal::Ciphertext>> Server::preciseSearch(
    const std::vector<std::vector<faiss::idx_t>> &nearest_coarse_vectors_id,
    const std::vector<seal::Ciphertext> &encrypted_precise_queries,
    seal::RelinKeys relin_keys, seal::GaloisKeys galois_keys) const {

    const float *dataset_base_ptr = m_DatasetBase.data();

    seal::SEALContext seal_ctx(m_EncryptionParms);
    seal::BatchEncoder batch_encoder(seal_ctx);
    seal::Evaluator evaluator(seal_ctx);

    std::vector<std::vector<seal::Ciphertext>> encrypted_precise_distances;
    encrypted_precise_distances.reserve(NQUERY);

    for (int i = 0; i < encrypted_precise_queries.size(); i++) {
        std::vector<seal::Ciphertext> coarse_probe_precise_distances;
        size_t coarse_probe = nearest_coarse_vectors_id[i].size();
        coarse_probe_precise_distances.reserve(coarse_probe);

        for (int j = 0; j < coarse_probe; j++) {
            const float *precise_vec_idx =
                dataset_base_ptr +
                (nearest_coarse_vectors_id[i][j] * PRECISE_VECTOR_DIMENSIONS);
            std::vector<int64_t> pod_vec(batch_encoder.slot_count(), 0);
            for (int k = 0; k < PRECISE_VECTOR_DIMENSIONS; k++) {
                pod_vec[k] = static_cast<int64_t>(precise_vec_idx[k]);
            }

            seal::Plaintext db_vec;
            batch_encoder.encode(pod_vec, db_vec);
            seal::Ciphertext sub_result;
            evaluator.sub_plain(encrypted_precise_queries[i], db_vec,
                                sub_result);

            // TODO: optimize the square here using the expansion formula.
            seal::Ciphertext result;
            evaluator.square(sub_result, result);
            evaluator.relinearize_inplace(result, relin_keys);

            for (size_t step = 1; step < PRECISE_VECTOR_DIMENSIONS;
                 step <<= 1) {
                seal::Ciphertext rotated;
                evaluator.rotate_rows(result, step, galois_keys, rotated);
                evaluator.add_inplace(result, rotated);
            }

            coarse_probe_precise_distances.push_back(result);
        }

        encrypted_precise_distances.push_back(coarse_probe_precise_distances);
    }

    return encrypted_precise_distances;
}

std::vector<std::vector<std::vector<seal::seal_byte>>>
Server::serialise_precise_search_results(
    const std::vector<std::vector<seal::Ciphertext>>
        &encrypted_precise_distances) const {

    std::vector<std::vector<std::vector<seal::seal_byte>>>
        encrypted_nquery_precise_distances;
    encrypted_nquery_precise_distances.reserve(
        encrypted_precise_distances.size());

    for (int i = 0; i < encrypted_precise_distances.size(); i++) {
        std::vector<std::vector<seal::seal_byte>>
            encrypted_coarse_probe_distances;
        encrypted_coarse_probe_distances.reserve(
            encrypted_precise_distances[i].size());

        for (int j = 0; j < encrypted_precise_distances[i].size(); j++) {
            std::vector<seal::seal_byte> serde_encrypted_precise_distances(
                encrypted_precise_distances[i][j].save_size());

            encrypted_precise_distances[i][j].save(
                serde_encrypted_precise_distances.data(),
                serde_encrypted_precise_distances.size());

            encrypted_coarse_probe_distances.push_back(
                serde_encrypted_precise_distances);
        }

        encrypted_nquery_precise_distances.push_back(
            encrypted_coarse_probe_distances);
    }

    return encrypted_nquery_precise_distances;
}

// TODO: Implement PIR for encrypted pipeline
// void Server::preciseVectorPIR(
//     const std::array<std::array<faiss_idx_t, K>, NQUERY>
//         &k_nearest_precise_vectors_idx,
//     std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
//                NQUERY> &query_results) {
//     // SPDLOG_INFO("Starting precise vector PIR on the server");
//
//     float *dataset_base_ptr = m_DatasetBase.data();
//
//     for (int i = 0; i < NQUERY; i++) {
//         for (int j = 0; j < K; j++) {
//             float *precise_vec_idx =
//                 dataset_base_ptr + (k_nearest_precise_vectors_idx[i][j] *
//                                     PRECISE_VECTOR_DIMENSIONS);
//
//             std::span<float> prec_vec(precise_vec_idx,
//                                       PRECISE_VECTOR_DIMENSIONS *
//                                           sizeof(precise_vec_idx));
//             for (int k = 0; k < PRECISE_VECTOR_DIMENSIONS; k++) {
//                 // SPDLOG_INFO("Vector {} Dimension k[{}] = {}", j, k,
//                 // prec_vec[k]);
//                 query_results[i][j][k] = prec_vec[k];
//             }
//         }
//     }
//
//     // SPDLOG_INFO("Precise vector PIR completed");
// }

// helper for debugging
void Server::display_nprobe_centroids(
    const std::vector<faiss::idx_t> &nprobe_centroids,
    size_t num_queries) const {

    size_t nprobe = nprobe_centroids.size() / num_queries;

    SPDLOG_INFO("\nDisplaying centroids, nprobe = {}\n", nprobe);
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < nprobe; j++) {
            printf("%lld, ", nprobe_centroids[i * nprobe + j]);
        }
        printf("\n");
    }
}
