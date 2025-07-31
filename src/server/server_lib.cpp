#include <cstdint>
#include <cstdio>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <seal/seal.h>
#include <spdlog/spdlog.h>

#include "client_server_utils.h"
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "server_lib.h"
#include "server_utils.h"

// Include controllers headers to register with server
#include "controllers/Query.h"

char const *SERVER_ADDRESS = "0.0.0.0";
constexpr int SERVER_PORT = 8080;

char const *TRAIN_DATASET_PATH = "sift/siftsmall/siftsmall_learn.fvecs";
char const *BASE_DATASET_PATH = "sift/siftsmall/siftsmall_base.fvecs";

char const *QUERY_DATASET_PATH = "sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "sift/siftsmall/siftsmall_groundtruth.ivecs";

// Path - build/_.faiss
std::string INDEX_FILE;

ServerEncryption::ServerEncryption(seal::EncryptionParameters encrypt_parms,
                                   const seal::SEALContext &seal_ctx)
    : EncryptedParms(std::move(encrypt_parms)), SealCtx(seal_ctx),
      BatchEncoder(seal_ctx) {}

Server::Server(size_t nlist, size_t sub_quantizers, size_t sub_quantizers_size,
               size_t poly_modulus, size_t plaintext_modulus,
               seal::EncryptionParameters &encrypt_params,
               seal::SEALContext &seal_ctx)
    : m_ServerEncryption(encrypt_params, seal_ctx) {

    m_Nlist = nlist;
    SubQuantizers = sub_quantizers;
    m_SubQuantizerSize = sub_quantizers_size;
    m_PolyModulusDegree = poly_modulus;
    m_PlaintextModulusSize = plaintext_modulus;

    std::ostringstream oss;
    oss << "build/" << "IVF" << m_Nlist << "_PQ" << SubQuantizers
        << "_SUB_QUANTIZER_SIZE" << m_SubQuantizerSize << ".faiss";
    INDEX_FILE = oss.str();
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
            &m_Quantizer, m_PreciseVectorDimensions, m_Nlist, SubQuantizers,
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
        SPDLOG_INFO("Reading cached data from index file = {}", INDEX_FILE);

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
    serde_parms.resize(
        static_cast<size_t>(m_ServerEncryption.EncryptedParms.save_size()));
    m_ServerEncryption.EncryptedParms.save(
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

    seal::RelinKeys relin_keys;
    relin_keys.load(m_ServerEncryption.SealCtx, serde_relin_keys.data(),
                    serde_relin_keys.size());

    seal::GaloisKeys galois_keys;
    galois_keys.load(m_ServerEncryption.SealCtx, serde_galois_keys.data(),
                     serde_galois_keys.size());

    // seal::SecretKey sk;
    // sk.load(m_ServerEncryption.SealCtx, enc_sk.data(), enc_sk.size());
    // seal::Decryptor decryptor(m_ServerEncryption.SealCtx, sk);

    std::vector<std::vector<seal::Ciphertext>>
        encrypted_nquery_residual_vectors;
    std::vector<std::vector<seal::Ciphertext>>
        encrypted_nquery_residual_vectors_squared;
    encrypted_nquery_residual_vectors.reserve(
        serde_encrypted_residual_vectors.size());
    encrypted_nquery_residual_vectors_squared.reserve(
        serde_encrypted_residual_vectors_squared.size());

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
                m_ServerEncryption.SealCtx,
                serde_encrypted_residual_vectors[i][j].data(),
                serde_encrypted_residual_vectors[i][j].size());

            encrypted_residual_vector_sq.load(
                m_ServerEncryption.SealCtx,
                serde_encrypted_residual_vectors_squared[i][j].data(),
                serde_encrypted_residual_vectors_squared[i][j].size());

            encrypted_query_residual_vectors.push_back(
                encrypted_residual_vector);
            encrypted_query_residual_vectors_squared.push_back(
                encrypted_residual_vector_sq);

            // seal::Plaintext decrypted_residual_vector;
            // seal::Plaintext decrypted_residual_vector_sq;
            // std::vector<int64_t> decoded_residual_vector;
            // std::vector<int64_t> decoded_residual_vector_squared;
            //
            // decryptor.decrypt(encrypted_residual_vector,
            //                   decrypted_residual_vector);
            // m_ServerEncryption.BatchEncoder.decode(decrypted_residual_vector,
            //                                        decoded_residual_vector);
            // std::for_each(decoded_residual_vector.begin(),
            //               decoded_residual_vector.end(),
            //               [](int64_t &n) { n /= BFV_SCALING_FACTOR; });
            //
            // decryptor.decrypt(encrypted_residual_vector_sq,
            //                   decrypted_residual_vector_sq);
            // std::vector<int64_t> encoded_u64_residual_vector_squared(1, 0LL);
            // m_ServerEncryption.BatchEncoder.decode(
            //     decrypted_residual_vector_sq,
            //     encoded_u64_residual_vector_squared);
            //
            // float decrypted_residual_vector_squared =
            //     encoded_u64_residual_vector_squared[0];
            // decrypted_residual_vector_squared /=
            //     (BFV_SCALING_FACTOR * BFV_SCALING_FACTOR);

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

std::pair<std::vector<std::vector<seal::Ciphertext>>,
          std::vector<std::vector<faiss::idx_t>>>
Server::coarseSearch(
    std::vector<faiss::idx_t> &nprobe_centroids,
    std::vector<std::vector<seal::Ciphertext>> &encrypted_residual_queries,
    std::vector<std::vector<seal::Ciphertext>>
        &encrypted_residual_queries_squared,
    size_t num_queries, size_t nprobe, seal::RelinKeys &relin_keys,
    seal::GaloisKeys &galois_keys) {

    m_Index->nprobe = nprobe;
    if (m_Index->nprobe != nprobe) {
        SPDLOG_ERROR("m_Index->nprobe != nprobe");
        throw std::runtime_error("m_Index->nprobe != nprobe");
    }

    seal::Evaluator evaluator(m_ServerEncryption.SealCtx);

    std::vector<std::vector<seal::Ciphertext>> encrypted_coarse_distances(
        num_queries * nprobe);
    std::vector<std::vector<faiss::idx_t>> coarse_distance_labels(num_queries *
                                                                  nprobe);

    m_Index->search_encrypted(
        m_ServerEncryption.BatchEncoder, evaluator, relin_keys, galois_keys,
        BFV_SCALING_FACTOR, num_queries, encrypted_residual_queries,
        encrypted_residual_queries_squared, nprobe_centroids.data(),
        encrypted_coarse_distances, coarse_distance_labels);

    return {encrypted_coarse_distances, coarse_distance_labels};
}

std::tuple<std::vector<seal::Ciphertext>, seal::RelinKeys, seal::GaloisKeys>
Server::deserialise_precise_search_params(
    const std::vector<std::vector<seal::seal_byte>>
        &serde_encrypted_precise_queries,
    const std::vector<seal::seal_byte> &serde_relin_keys,
    const std::vector<seal::seal_byte> &serde_galois_keys) const {

    seal::RelinKeys relin_keys;
    relin_keys.load(m_ServerEncryption.SealCtx, serde_relin_keys.data(),
                    serde_relin_keys.size());

    seal::GaloisKeys galois_keys;
    galois_keys.load(m_ServerEncryption.SealCtx, serde_galois_keys.data(),
                     serde_galois_keys.size());

    std::vector<seal::Ciphertext> encrypted_precise_queries;
    encrypted_precise_queries.reserve(serde_encrypted_precise_queries.size());

    for (int i = 0; i < serde_encrypted_precise_queries.size(); i++) {
        seal::Ciphertext encrypted_precise_query;

        encrypted_precise_query.load(m_ServerEncryption.SealCtx,
                                     serde_encrypted_precise_queries[i].data(),
                                     serde_encrypted_precise_queries[i].size());

        encrypted_precise_queries.push_back(encrypted_precise_query);
    }

    return {encrypted_precise_queries, relin_keys, galois_keys};
}

std::vector<std::vector<seal::Ciphertext>> Server::preciseSearch(
    const std::vector<std::vector<faiss::idx_t>> &nearest_coarse_vectors_id,
    const std::vector<seal::Ciphertext> &encrypted_precise_queries,
    const seal::RelinKeys &relin_keys, const seal::GaloisKeys &galois_keys) {

    const float *dataset_base_ptr = m_DatasetBase.data();

    seal::Evaluator evaluator(m_ServerEncryption.SealCtx);

    std::vector<std::vector<seal::Ciphertext>> encrypted_precise_distances;
    encrypted_precise_distances.reserve(encrypted_precise_queries.size());

    for (int i = 0; i < encrypted_precise_queries.size(); i++) {
        std::vector<seal::Ciphertext> coarse_probe_precise_distances;
        size_t coarse_probe = nearest_coarse_vectors_id[i].size();
        coarse_probe_precise_distances.reserve(coarse_probe);

        for (int j = 0; j < coarse_probe; j++) {
            const float *precise_vec_idx =
                dataset_base_ptr +
                (nearest_coarse_vectors_id[i][j] * m_PreciseVectorDimensions);
            std::vector<int64_t> pod_vec(
                m_ServerEncryption.BatchEncoder.slot_count(), 0);
            for (int k = 0; k < m_PreciseVectorDimensions; k++) {
                pod_vec[k] = static_cast<int64_t>(precise_vec_idx[k]);
            }

            seal::Plaintext db_vec;
            m_ServerEncryption.BatchEncoder.encode(pod_vec, db_vec);

            seal::Ciphertext result = L2sqr_encrypted(
                evaluator, m_ServerEncryption.BatchEncoder, db_vec,
                encrypted_precise_queries[i], relin_keys, galois_keys,
                m_PreciseVectorDimensions);
            coarse_probe_precise_distances.push_back(result);
        }

        encrypted_precise_distances.push_back(coarse_probe_precise_distances);
    }

    return encrypted_precise_distances;
}

std::vector<std::vector<std::vector<seal::seal_byte>>>
Server::serialise_encrypted_distances(
    const std::vector<std::vector<seal::Ciphertext>> &encrypted_distances)
    const {

    std::vector<std::vector<std::vector<seal::seal_byte>>>
        serde_nquery_encrypted_distances;
    serde_nquery_encrypted_distances.reserve(encrypted_distances.size());

    for (int i = 0; i < encrypted_distances.size(); i++) {
        std::vector<std::vector<seal::seal_byte>>
            serde_per_query_encrypted_distances;
        serde_per_query_encrypted_distances.reserve(
            encrypted_distances[i].size());

        for (int j = 0; j < encrypted_distances[i].size(); j++) {
            std::vector<seal::seal_byte> serde_encrypted_coarse_distances(
                encrypted_distances[i][j].save_size());

            encrypted_distances[i][j].save(
                serde_encrypted_coarse_distances.data(),
                serde_encrypted_coarse_distances.size());

            serde_per_query_encrypted_distances.push_back(
                serde_encrypted_coarse_distances);
        }

        serde_nquery_encrypted_distances.push_back(
            serde_per_query_encrypted_distances);
    }

    return serde_nquery_encrypted_distances;
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
    const size_t num_queries) const {

    size_t nprobe = nprobe_centroids.size() / num_queries;

    SPDLOG_INFO("\nDisplaying centroids, nprobe = {}\n", nprobe);
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < nprobe; j++) {
            printf("%lld, ", nprobe_centroids[i * nprobe + j]);
        }
        printf("\n");
    }
}

// -----------------------------------
// Single Phase Search

std::tuple<std::vector<seal::Ciphertext>, seal::RelinKeys, seal::GaloisKeys>
Server::deserialise_single_phase_search_parms(
    const std::vector<std::vector<seal::seal_byte>>
        &serde_encrypted_query_vectors,
    const std::vector<seal::seal_byte> &serde_relin_keys,
    const std::vector<seal::seal_byte> &serde_galois_keys) const {

    seal::RelinKeys relin_keys;
    relin_keys.load(m_ServerEncryption.SealCtx, serde_relin_keys.data(),
                    serde_relin_keys.size());

    seal::GaloisKeys galois_keys;
    galois_keys.load(m_ServerEncryption.SealCtx, serde_galois_keys.data(),
                     serde_galois_keys.size());

    std::vector<seal::Ciphertext> nquery_encrypted_vectors;
    nquery_encrypted_vectors.reserve(serde_encrypted_query_vectors.size());

    for (int i = 0; i < serde_encrypted_query_vectors.size(); i++) {

        seal::Ciphertext encrypted_query_vector;
        encrypted_query_vector.load(m_ServerEncryption.SealCtx,
                                    serde_encrypted_query_vectors[i].data(),
                                    serde_encrypted_query_vectors[i].size());

        nquery_encrypted_vectors.push_back(encrypted_query_vector);
    }

    return {nquery_encrypted_vectors, relin_keys, galois_keys};
}

std::pair<std::vector<std::vector<seal::Ciphertext>>,
          std::vector<std::vector<faiss::idx_t>>>
Server::singlePhaseSearch(
    const std::vector<faiss::idx_t> &nprobe_centroids,
    const std::vector<seal::Ciphertext> &encrypted_queries,
    const size_t num_queries, const size_t nprobe,
    const seal::RelinKeys &relin_keys, const seal::GaloisKeys &galois_keys) {

    seal::Evaluator evaluator(m_ServerEncryption.SealCtx);

    const float *dataset_base_ptr = m_DatasetBase.data();
    std::vector<size_t> list_sizes_per_query;
    list_sizes_per_query.reserve(num_queries);

    for (int64_t i = 0; i < num_queries; ++i) {
        size_t total = 0;
        for (int64_t j = 0; j < nprobe; ++j) {
            int64_t key = nprobe_centroids[i * nprobe + j];
            total += m_Index->invlists->list_size(key);
        }
        list_sizes_per_query.push_back(total);
    }

    std::vector<std::vector<seal::Ciphertext>> encrypted_distances;
    encrypted_distances.reserve(num_queries);
    std::vector<std::vector<faiss::idx_t>> labels_nquery;
    labels_nquery.reserve(num_queries);

    for (int i = 0; i < num_queries; i++) {

        std::vector<seal::Ciphertext> encrypted_distance;
        encrypted_distance.reserve(list_sizes_per_query[i]);
        std::vector<faiss::idx_t> labels;
        labels.reserve(list_sizes_per_query[i]);

        for (int j = 0; j < nprobe; j++) {
            const size_t code_size =
                m_Index->invlists->list_size(nprobe_centroids[i * nprobe + j]);

            for (int k = 0; k < code_size; k++) {
                seal::Ciphertext result;
                size_t db_index = m_Index->invlists->get_ids(
                    nprobe_centroids[i * nprobe + j])[k];
                const float *db_query =
                    dataset_base_ptr + (db_index * m_PreciseVectorDimensions);

                // get plain text vector
                std::vector<int64_t> pod_vec(
                    m_ServerEncryption.BatchEncoder.slot_count(), 0);
                for (int l = 0; l < m_PreciseVectorDimensions; l++) {
                    pod_vec[l] = static_cast<int64_t>(db_query[l]);
                }
                seal::Plaintext db_vec;
                m_ServerEncryption.BatchEncoder.encode(pod_vec, db_vec);

                // compute l2 distance
                result =
                    L2sqr_encrypted(evaluator, m_ServerEncryption.BatchEncoder,
                                    db_vec, encrypted_queries[i], relin_keys,
                                    galois_keys, m_PreciseVectorDimensions);

                // push distance into vector
                encrypted_distance.push_back(result);
                labels.push_back(db_index);
            }
        }

        encrypted_distances.push_back(encrypted_distance);
        labels_nquery.push_back(labels);
    }

    return {encrypted_distances, labels_nquery};
}
