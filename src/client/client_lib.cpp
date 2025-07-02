#include "spdlog/spdlog.h"
#include "cpr/cpr.h"
#include "json/json.h"
#include "client_lib.h"
#include "faiss/Index.h"
#include "faiss/MetricType.h"
#include <cassert>
#include <cstddef>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVF.h>
#include <stdexcept>
#include <sys/time.h>
#include <sys/stat.h>

int add_dist(size_t index, size_t nprobe, float dist, float *temp_nprobe_store, faiss::idx_t* temp_nprobe_store_indexes) {
    for (int i = 0; i < nprobe; i++) {
      if (dist < temp_nprobe_store[i]) {
          for (int j = nprobe - 1; j > i; j--) {
              temp_nprobe_store[j] = temp_nprobe_store[j - 1];
          }
          for (int j = nprobe - 1; j > i; j--) {
              temp_nprobe_store_indexes[j] = temp_nprobe_store_indexes[j - 1];
          }
          temp_nprobe_store[i] = dist;
          temp_nprobe_store_indexes[i] = index;
          return 1;
      }
  }
  return 0;
}

void search_centroids(size_t nprobe, float* x_query, idx_t* centroid_indexes) {
    auto compute_distance = [&] (float* centroid_value, float* query) {
        size_t i;
        float res = 0;
        for (i = 0; i < d; i++) {
            const float tmp = centroid_value[i] - query[i];
            res += tmp * tmp;
        }
        return res;
    };

    for (int ith_query = 0; ith_query < n_query; ith_query++) {
        faiss::idx_t *nprobe_store_indexes = new faiss::idx_t[nprobe];
        float *nprobe_store = new float[nprobe];
        for (int i = 0; i<nprobe; i++){
          nprobe_store[i] = 1e10;
        }
        for (int j = 0; j<4096; j++) {
          float dist = compute_distance(&centroid_values[j*d], &x_query[ith_query*d]);
          add_dist(j, nprobe, dist, nprobe_store, nprobe_store_indexes);
        }

        for (int i = 0; i<nprobe; i++){
          assert(nprobe_store_indexes[i] < 4096);
          centroid_indexes[ith_query * nprobe + i] = nprobe_store_indexes[i];
        }
    }
    
    // temp_nprobe_store_indexes contains the selected nprobe indexes.
    return;
}

void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url("http://localhost:8080/ping"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);
}

void get_centroids() {
    SPDLOG_INFO("Sending a request to /query at localhost 8080");
    cpr::Response r = cpr::Get(cpr::Url("http://localhost:8080/query"));
    SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);
}
