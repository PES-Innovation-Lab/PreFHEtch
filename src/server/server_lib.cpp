#include <cstdio>
#include <vector>

#include <faiss/index_io.h>
#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_server_utils.h"
#include "server_lib.h"

// Include controllers headers to register with server
#include "controllers/Query.h"
#include "faiss/MetricType.h"

char const *SERVER_ADDRESS = "localhost";

char const *TRAIN_DATASET_PATH = "../sift/siftsmall/siftsmall_learn.fvecs";
char const *BASE_DATASET_PATH = "../sift/siftsmall/siftsmall_base.fvecs";

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

Server::Server()
    : m_Quantizer(PRECISE_VECTOR_DIMENSIONS),
      m_Index(std::make_unique<faiss::IndexIVFPQ>(&m_Quantizer, PRECISE_VECTOR_DIMENSIONS, NLIST, SUB_QUANTIZERS,
              SUB_VECTOR_SIZE)) {
    SPDLOG_INFO("Preparing index with precise dimension d={}",
                PRECISE_VECTOR_DIMENSIONS);
}

void Server::run_webserver() {
    drogon::app().addListener(SERVER_ADDRESS, SERVER_PORT);

    SPDLOG_INFO("Server listening on {}:8080", SERVER_ADDRESS);
    drogon::app().run();
}

void Server::init_index() {
    // IVF with 128 centroids gives R@10 of 0.86 for sift10k
    size_t d;

    // Training the index
    if (!(std::filesystem::exists("test.faiss"))){
          SPDLOG_INFO("Loading train set");

          size_t nt;
          std::vector<float> xt;
          vecs_read<float>(TRAIN_DATASET_PATH, d, nt, xt);

          if (d != PRECISE_VECTOR_DIMENSIONS) {
              throw std::runtime_error("Incorrect dimensions for train set, not "
                                       "the same as PRECISE_VECTOR_DIMENSIONS");
          }

          SPDLOG_INFO("Training on {} vectors", nt);
          m_Index->train(nt, xt.data());

          SPDLOG_INFO("Loading database");

          size_t nb, d2;
          vecs_read<float>(BASE_DATASET_PATH, d2, nb, m_DatasetBase);
          assert(d == d2 || !"dataset does not have same dimension as train set");

          SPDLOG_INFO("Indexing database, size {}*{}", nb, d);
          m_Index->add(nb, m_DatasetBase.data());
      
      faiss::write_index(m_Index.get(), "test.faiss");
      SPDLOG_INFO("Cached dataset");

    } else {
        SPDLOG_INFO("Reading cached data");

        faiss::Index* loaded_ptr = faiss::read_index("test.faiss");
        auto* loaded_ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(loaded_ptr);
        if (!loaded_ivfpq) {
            throw std::runtime_error("Loaded index is not of type IndexIVFPQ");
        }

        m_Index.reset(loaded_ivfpq);
        d = m_Index->d;
        
        // delete loaded_ivfpq;
    }

    size_t nq;
    std::vector<float> xq;

    {
        SPDLOG_INFO("Loading queries");

        size_t d2;
        vecs_read<float>(QUERY_DATASET_PATH, d2, nq, xq);
        assert(d == d2 || !"query does not have same dimension as train set");

        for (int i = 0; i < NQUERY; i++) {
            printf("Query i = %d", i + 1);
            for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++) {
                size_t idx = i * PRECISE_VECTOR_DIMENSIONS + j;
                printf("(%d) = %f, ", idx, xq[idx]);
            }
            printf("\n");
        }
    }

    size_t k; // nb of results per query in the GT
    std::vector<faiss::idx_t>
        gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        SPDLOG_INFO("Loading ground truth for {} queries", nq);

        // load ground-truth and convert int to long
        size_t nq2;
        std::vector<int> gt_int;
        vecs_read<int>(GROUNDTRUTH_DATASET_PATH, k, nq2, gt_int);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt.resize(k * nq);
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }

    }

    // Use the found configuration to perform a search
    {
        SPDLOG_INFO("Perform a search on {} queries", nq);

        // output buffers
        faiss::idx_t *I = new faiss::idx_t[nq * k];
        float *D = new float[nq * k];

        m_Index->search(nq, xq.data(), k, D, I);

        int temp_k = 10;
        SPDLOG_INFO("Ground truth results of query[0] ({} nearest-neighbours):",
                    temp_k);
        for (int i = 0; i < temp_k; i++) {
            printf("k@%d = %lld, ", i + 1, gt[i]);
        }
        printf("\n");

        SPDLOG_INFO("Calculated results of query[0] ({} nearest-neighbours):",
                    temp_k);
        for (int i = 0; i < temp_k; i++) {
            printf("k@%d = %lld, ", i + 1, I[i]);
        }
        printf("\n");

        SPDLOG_INFO("Compute recalls");

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            long long gt_nn = gt[i * k];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

        delete[] I;
        delete[] D;
    }
}

void Server::retrieve_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids) {
    centroids.resize(NLIST);
    for (int i = 0; i < NLIST; i++) {
        m_Quantizer.reconstruct(i, centroids[i].data());
    }
}

void Server::coarseSearch(
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    const std::array<std::array<faiss::idx_t, NPROBE>, NQUERY>
        &nearest_centroid_idx,
    std::vector<float> &coarse_distance_scores,
    std::vector<faiss::idx_t> &coarse_distance_indexes,
    std::array<size_t, NQUERY> &list_sizes_per_query) {

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

    SPDLOG_INFO("Coarse search complete");
}

void Server::preciseSearch(
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    const std::array<std::array<faiss::idx_t, COARSE_PROBE>, NQUERY>
        &nearest_coarse_vector_idx,
    std::array<std::array<float, COARSE_PROBE>, NQUERY>
        &precise_distance_scores) {
    SPDLOG_INFO("Starting precise search on the server");

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

    SPDLOG_INFO("Precise search completed");
}

void Server::preciseVectorPIR(
    const std::array<std::array<faiss_idx_t, K>, NQUERY>
        &k_nearest_precise_vectors_idx,
    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY> &query_results) {
    SPDLOG_INFO("Starting precise vector PIR on the server");

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

    SPDLOG_INFO("Precise vector PIR completed");
}
