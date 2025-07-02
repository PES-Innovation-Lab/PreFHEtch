#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

// Dataset - SIFT10K

constexpr int64_t PRECISE_VECTOR_DIMENSIONS = 128;
constexpr int64_t COARSE_VECTOR_DIMENSIONS = 8;

constexpr int64_t NPROBE = 5;
constexpr int64_t K = 100;
constexpr int64_t NLIST = 256;
constexpr int64_t SUB_QUANTIZERS = 8;
constexpr int64_t SUB_VECTOR_SIZE = 8;

// Singleton class pattern for static access across all controllers
class Server {
  private:
    faiss::IndexFlatL2 m_Quantizer;
    faiss::IndexIVFPQ m_Index;

  public:
    Server();
    static std::shared_ptr<Server> &getInstance() {
        static std::shared_ptr<Server> srvr = std::make_shared<Server>();
        return srvr;
    }

    void init_index(const char *index_key);
    void run_webserver();
    void retrieve_centroids(
        std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids);
};
