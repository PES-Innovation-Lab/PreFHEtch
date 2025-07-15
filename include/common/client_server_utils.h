#pragma once

#include <sys/stat.h>

#include <spdlog/spdlog.h>

// Dataset - SIFT10K

constexpr int64_t PRECISE_VECTOR_DIMENSIONS = 128;
constexpr int64_t COARSE_VECTOR_DIMENSIONS = 8;

constexpr int64_t NPROBE = 20;
constexpr int64_t COARSE_PROBE = 200;
constexpr int64_t K = 100;
constexpr int64_t NBASE = 10000;
constexpr int64_t NQUERY = 5;

// TODO: (nbase/nprobe) * nq
constexpr int64_t NLIST = 256;
constexpr int64_t SUB_QUANTIZERS = 32;
constexpr int64_t SUB_VECTOR_SIZE = 8;

using faiss_idx_t = int64_t;

template <typename T>
void vecs_read(const char *fname, size_t &d_out, size_t &n_out,
               std::vector<T> &vecs) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        SPDLOG_ERROR("could not open {}", fname);
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
                d * sizeof(T));

    fclose(f);
}
