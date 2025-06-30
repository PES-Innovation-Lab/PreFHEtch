#include <cstdio>
#include <vector>

#include <sys/stat.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "server_lib.h"

// Include controllers headers to register with server
#include "controllers/Query.h"

char const *TRAIN_DATASET_PATH = "../sift/siftsmall/siftsmall_learn.fvecs";
char const *BASE_DATASET_PATH = "../sift/siftsmall/siftsmall_base.fvecs";

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

void init_logger() {}

void run_webserver() {
    init_logger();
    drogon::app().addListener("localhost", 8080);

    SPDLOG_INFO("Server listening on localhost:8080");
    drogon::app().run();
}

void init_index() {
    const char *index_key = "IVF4096,Flat";
    faiss::Index *index;
    size_t d;

    // Training the index
    {
        SPDLOG_INFO("Loading train set");

        size_t nt;
        std::vector<float> xt;
        fvecs_read(TRAIN_DATASET_PATH, d, nt, xt);

        SPDLOG_INFO("Preparing index \"{}\" d={}", index_key, d);
        index = faiss::index_factory(d, index_key);

        SPDLOG_INFO("Training on {} vectors", nt);
        index->train(nt, xt.data());
    }

    // Adding vectors to the index
    {
        SPDLOG_INFO("Loading database");

        size_t nb, d2;
        std::vector<float> xb;
        fvecs_read(BASE_DATASET_PATH, d2, nb, xb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        SPDLOG_INFO("Indexing database, size {}*{}", nb, d);
        index->add(nb, xb.data());
    }

    size_t nq;
    std::vector<float> xq;

    {
        SPDLOG_INFO("Loading queries");

        size_t d2;
        fvecs_read(QUERY_DATASET_PATH, d2, nq, xq);
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
        ivecs_read(GROUNDTRUTH_DATASET_PATH, k, nq2, gt_int);
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
        params.initialize(index);

        SPDLOG_INFO("Auto-tuning over {} parameters ({} combinations)",
                    params.parameter_ranges.size(), params.n_combinations());

        faiss::OperatingPoints ops;
        params.explore(index, nq, xq.data(), crit, &ops);

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
        params.set_index_parameters(index, selected_params.c_str());

        SPDLOG_INFO("Perform a search on {} queries", nq);

        // output buffers
        faiss::idx_t *I = new faiss::idx_t[nq * k];
        float *D = new float[nq * k];

        index->search(nq, xq.data(), k, D, I);

        SPDLOG_INFO("Compute recalls");

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
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

// Returns a dummy set of NUM_CENTROIDS centroids between 0 and 1
void retrieve_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids) {
    centroids.reserve(NUM_CENTROIDS);

    for (int i = 0; i < NUM_CENTROIDS; i++) {
        std::array<float, PRECISE_VECTOR_DIMENSIONS> centroid;

        float start = 0.0;
        constexpr float step = 1.0 / PRECISE_VECTOR_DIMENSIONS;
        for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++, start += step) {
            centroid[j] = start;
        }

        centroids.push_back(centroid);
    }
}

void fvecs_read(const char *fname, size_t &d_out, size_t &n_out,
                std::vector<float> &vecs) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        SPDLOG_ERROR("could not open %s", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"Incorrect dimensions");
    fseek(f, 0, SEEK_SET);
    struct stat st{};
    fstat(fileno(f), &st);
    const size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"Incorrect file size");
    const size_t n = sz / ((d + 1) * 4);

    d_out = d;
    n_out = n;
    vecs.resize(n * (d + 1));
    const size_t nr = fread(vecs.data(), sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(vecs.data() + i * d, vecs.data() + 1 + i * (d + 1),
                d * sizeof(float));

    fclose(f);
}

void ivecs_read(const char *fname, size_t &d_out, size_t &n_out,
                std::vector<int> &vecs) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        SPDLOG_ERROR("could not open %s", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"Incorrect dimensions");
    fseek(f, 0, SEEK_SET);
    struct stat st{};
    fstat(fileno(f), &st);
    const size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"Incorrect file size");
    const size_t n = sz / ((d + 1) * 4);

    d_out = d;
    n_out = n;
    vecs.resize(n * d);
    int *tmp = new int[n * (d + 1)];
    fread(tmp, sizeof(int), n * (d + 1), f);
    for (size_t i = 0; i < n; i++) {
        memcpy(vecs.data() + i * d, tmp + 1 + i * (d + 1), d * sizeof(int));
    }
    delete[] tmp;
    fclose(f);
}
