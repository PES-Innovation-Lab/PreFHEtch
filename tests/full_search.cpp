// Just take a high dimension vector and convert it into low dimension using PQ
// should use faiss ofcourse
#include "faiss/Index.h"
#include "faiss/MetricType.h"
#include <cassert>
#include <cstddef>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVF.h>

#include <stdexcept>
#include <limits.h>
#include <sys/time.h>
#include <sys/stat.h>


void sort_floats_with_labels(float* arr, faiss::idx_t* labels, size_t n) {
    for (size_t i = 0; i < n - 1; i++) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }

        // Swap values
        if (min_idx != i) {
            float temp_val = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp_val;

            int temp_label = labels[i];
            labels[i] = labels[min_idx];
            labels[min_idx] = temp_label;
        }
    }
}

namespace faiss {

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d; // Dimensions.
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st); 
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4); // Number of vectors.

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)]; // Raw header buffer containing the vectors.
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

}  // namespace faiss

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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


int main() {
  
  double t0 = elapsed();
  
  size_t n_learn;
  size_t d;
  size_t n_base;

  faiss::Index* index;


  size_t n_query;
  size_t n_gt;
  size_t k;
 
  float *x_learn = faiss::fvecs_read("./dataset/sift1M/sift_learn.fvecs", &d, &n_learn);
  printf("n_learn : %ld, d : %ld\n", n_learn, d);
  
  float *x_base = faiss::fvecs_read("./dataset/sift1M/sift_base.fvecs", &d, &n_base);
  printf("n_base : %ld\n", n_base);

  int *gt_int = (int *) faiss::fvecs_read("./dataset/sift1M/sift_groundtruth.ivecs", &k, &n_gt);
  float *x_query = faiss::fvecs_read("./dataset/sift1M/sift_query.fvecs", &d, &n_query);
  printf("k : %ld\n", k);
  index = faiss::index_factory(d, "IVF4096,PQ8");

  auto *ivf = dynamic_cast<faiss::IndexIVF*>(index);
  if (ivf){
    // do nothing.
  } else throw std::runtime_error("Error casting index to IVF index.");

  printf("[%.3f s] Starting Training\n", elapsed()-t0);
  ivf->train(n_learn, x_learn);

  delete[] x_learn;

  printf("[%.3f s] Adding vectors into database\n", elapsed()-t0);
  ivf->add(n_base, x_base);
  
  delete[] x_base;

  printf("[%.3f s] Finished Adding vectors to the database\n", elapsed()-t0);

  size_t nprobe = 20;
  faiss::idx_t* labels = new faiss::idx_t[n_query * nprobe * 250];
  float* distances = new float[n_query * nprobe * 250];
  faiss::idx_t* centroid_indexes = new faiss::idx_t[n_query*nprobe];

  // get the centroid indexes from the client 
  // we will send the client the data using index->get_centroids();
  // for now lets choose all the centroids to search through
  
  // these are the full precision vectors.
  // need to search through them.
  float* centroid_values = ivf->get_IVF_centroids();

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
    // compute nprobe number of centroids.
    // go through centroid_values, compute distance.
    faiss::idx_t *temp_nprobe_store_indexes = new faiss::idx_t[nprobe];
    float *temp_nprobe_store = new float[nprobe];
    for (int i = 0; i<nprobe; i++){
      temp_nprobe_store[i] = 1e10;
    }
    for (int j = 0; j<4096; j++) {
      float dist = compute_distance(&centroid_values[j*d], &x_query[ith_query*d]);
      add_dist(j, nprobe, dist, temp_nprobe_store, temp_nprobe_store_indexes);
    }

    for (int i = 0; i<nprobe; i++){
      assert(temp_nprobe_store_indexes[i] < 4096);
      centroid_indexes[ith_query * nprobe + i] = temp_nprobe_store_indexes[i];
    }
  }

  size_t* list_sizes_per_query = new size_t[n_query];

  auto searchtime = elapsed();
  // ivf->search(n_query, x_query, k, distances, labels);
  ivf->search_encrypted(n_query, x_query,centroid_indexes, distances, labels, list_sizes_per_query);
  searchtime = elapsed() - searchtime;

  printf("Search Time : %f s\n", searchtime);

  delete[] centroid_indexes;

  faiss::idx_t offset = 0;
  for (faiss::idx_t i = 0; i<n_query; i++) {
    // printf("%ld query\n", i);
    for (size_t j = 0; j < list_sizes_per_query[i]; j++) {
      // printf("Distance : %f, Label : %ld\n", distances[offset], labels[offset]);
      offset++;
    }
  }

  faiss::idx_t to_add = 0;
  for (int i = 0; i<n_query; i++) {
    sort_floats_with_labels(&distances[to_add], &labels[to_add], list_sizes_per_query[i]);
    to_add += list_sizes_per_query[i];
  }

  delete[] distances;

  faiss::idx_t* gt = new faiss::idx_t[k * n_query];
  for (int i = 0; i < k * n_query; i++) {
      gt[i] = gt_int[i];
  }

  to_add = 0;
  int k_limit = 15;
  int* n_k = new int[k_limit];
  for (int i = 0; i<k_limit; i++){
    n_k[i] = 0;
  }
  int n_1 = 0, n_10 = 0, n_100 = 0, n_max = 0;
  for (int i = 0; i < n_query; i++) {
      // for each query find the nearest neighbour.
      int gt_nn = gt[i * k];
      faiss::idx_t query_list_size = list_sizes_per_query[i];
      for (int j = 0; j < query_list_size; j++) {
          // check if the query is within 
          // 1, 10 or 100.
          if (labels[to_add + j] == gt_nn) {
              if (j < 1)
                  n_1++;
              if (j < 10)
                  n_10++;
              if (j < 100)
                  n_100++;
              if (j < k_limit)
                  n_k[j]++;
          }
      }
      to_add += query_list_size;
  }


  printf("R@1 = %.4f\n", n_1 / float(n_query));
  printf("R@10 = %.4f\n", n_10 / float(n_query));
  printf("R@100 = %.4f\n", n_100 / float(n_query));

  float prev_output = 0;
  for (int i = 0; i<k_limit; i++) {
    prev_output +=  n_k[i]/float(n_query); 
    printf("R@%d:%.4f\n", i+1, prev_output);
  }

  // Fine search
  // Fine search only the top k = 100 vectors.
  // The recall computed after this is for R@1
  
  x_base = faiss::fvecs_read("./dataset/sift1M/sift_base.fvecs", &d, &n_base);
  float finesearchtime = elapsed();
  faiss::idx_t* closest_labels = new faiss::idx_t[n_query];

  to_add = 0;

  k = 100;
  for (int i = 0; i < n_query; i++) {
      float* query = &x_query[i * d];

      float min_dist = std::numeric_limits<float>::max();
      faiss::idx_t best_label = -1;

      for (int j = 0; j < k; j++) {
          faiss::idx_t db_idx = labels[to_add + j];
          float* db_vector = &x_base[db_idx * d];

          float sum = 0.0f;
          for (int l = 0; l < d; l++) {
              float d0 = query[l] - db_vector[l];
              sum += d0 * d0;
          }

          if (sum < min_dist) {
              min_dist = sum;
              best_label = db_idx;
          }
      }

      closest_labels[i] = best_label;
      to_add += list_sizes_per_query[i];
  }
  
  n_1 = 0;
  for (int i = 0; i < n_query; i++){
    if (closest_labels[i] == gt_int[i*k]){
      n_1++;
    }
  }

  printf("finesearchtime : %f\n", elapsed() - finesearchtime);
  printf("R@1 IF WE DO FINE SEARCH THROUGH ALL THE SELECTED VECTORS : %f\n", n_1/float(n_query));

  printf("[%.3f s] total number of vectors : %ld\n",elapsed()-t0,  offset);
}
