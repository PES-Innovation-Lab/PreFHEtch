#include <cstdio>
#include <vector>

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

char const *TRAIN_DATASET_PATH = "../sift/siftsmall/siftsmall_learn.fvecs";
char const *BASE_DATASET_PATH = "../sift/siftsmall/siftsmall_base.fvecs";

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

Server::Server()
    : m_Quantizer(PRECISE_VECTOR_DIMENSIONS),
      m_Index(&m_Quantizer, PRECISE_VECTOR_DIMENSIONS, NLIST, SUB_QUANTIZERS,
              SUB_VECTOR_SIZE) {
    SPDLOG_INFO("Preparing index with precise dimension d={}",
                PRECISE_VECTOR_DIMENSIONS);
}

void Server::run_webserver() {
    drogon::app().addListener("localhost", 8080);

    SPDLOG_INFO("Server listening on localhost:8080");
    drogon::app().run();
}

void Server::init_index() {
    // IVF with 128 centroids gives R@10 of 0.86 for sift10k
    size_t d;

    // Training the index
    {
        SPDLOG_INFO("Loading train set");

        size_t nt;
        std::vector<float> xt;
        vecs_read<float>(TRAIN_DATASET_PATH, d, nt, xt);

        SPDLOG_INFO("Training on {} vectors", nt);
        m_Index.train(nt, xt.data());
    }

    // Adding vectors to the index
    {
        SPDLOG_INFO("Loading database");

        size_t nb, d2;
        std::vector<float> xb;
        vecs_read<float>(BASE_DATASET_PATH, d2, nb, xb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        SPDLOG_INFO("Indexing database, size {}*{}", nb, d);
        m_Index.add(nb, xb.data());
    }

    size_t nq;
    std::vector<float> xq;

    {
        SPDLOG_INFO("Loading queries");

        size_t d2;
        vecs_read<float>(QUERY_DATASET_PATH, d2, nq, xq);
        assert(d == d2 || !"query does not have same dimension as train set");
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

        // std::ranges::transform(gt_int, gt.begin(),
        //                        [](const int d) { return faiss::idx_t(d); });
    }

    // Result of the auto-tuning
    std::string selected_params;

    // Run auto-tuning
    {
        SPDLOG_INFO("Preparing auto-tune criterion 1-recall at 1 "
                    "criterion, with k=%ld nq=%ld",
                    k, nq);

        faiss::OneRecallAtRCriterion crit(nq, 1);
        crit.set_groundtruth(k, nullptr, gt.data());
        crit.nnn = k; // by default, the criterion will request only 1 NN

        SPDLOG_INFO("Preparing auto-tune parameters");

        faiss::ParameterSpace params;
        params.initialize(&m_Index);

        SPDLOG_INFO("Auto-tuning over {} parameters ({} combinations)",
                    params.parameter_ranges.size(), params.n_combinations());

        faiss::OperatingPoints ops;
        params.explore(&m_Index, nq, xq.data(), crit, &ops);

        SPDLOG_INFO("Found the following operating points: ");
        ops.display();

        // keep the first parameter that obtains > 0.5 1-recall@1
        for (int i = 0; i < ops.optimal_pts.size(); i++) {
            if (ops.optimal_pts[i].perf > 0.5) {
                selected_params = ops.optimal_pts[i].key;
                break;
            }
        }
        assert(selected_params.size() >= 0 ||
               !"could not find good enough op point");
    }

    // Use the found configuration to perform a search
    {
        faiss::ParameterSpace params;

        SPDLOG_INFO("Setting parameter configuration \"{}\" on index",
                    selected_params.c_str());
        params.set_index_parameters(&m_Index, selected_params.c_str());

        SPDLOG_INFO("Perform a search on {} queries", nq);

        // output buffers
        faiss::idx_t *I = new faiss::idx_t[nq * k];
        float *D = new float[nq * k];

        m_Index.search(nq, xq.data(), k, D, I);

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

void Server::prefilter(
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::array<int64_t, NPROBE> &nearest_centroid_idx,
    std::vector<float> &coarse_distance_scores,
    std::vector<faiss::idx_t> &coarse_distance_indexes,
    std::array<size_t, NQUERY> &list_sizes_per_query) {

    // Reset nprobe, previously set by auto-tuning
    m_Index.nprobe = NPROBE;

    coarse_distance_scores.resize(NBASE * NQUERY);
    coarse_distance_indexes.resize(NBASE * NQUERY);

    m_Index.search_encrypted(
        1, precise_query.data(), nearest_centroid_idx.data(),
        coarse_distance_scores.data(), coarse_distance_indexes.data(),
        list_sizes_per_query.data());

    size_t coarse_vectors_count = std::accumulate(
        list_sizes_per_query.begin(), list_sizes_per_query.end(), 0);
    coarse_distance_scores.resize(coarse_vectors_count);
    coarse_distance_indexes.resize(coarse_vectors_count);

    SPDLOG_INFO("Prefiltering complete");
}
