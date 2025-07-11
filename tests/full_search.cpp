#include <filesystem>
// Just take a high dimension vector and convert it into low dimension using PQ
// should use faiss ofcourse
#include "faiss/Index.h"
#include <faiss/index_io.h>
#include "faiss/MetricType.h"
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVF.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <vector>

// const char* dataset_learn = "./sift/sift1M/sift_learn.fvecs";
// const char* dataset_base = "./sift/sift1M/sift_base.fvecs";
// const char* dataset_query = "./sift/sift1M/sift_query.fvecs";
// const char* dataset_gt = "./sift/sift1M/sift_groundtruth.ivecs";
//
// const char* index_key = "IVF4096,PQ8";
// const size_t nprobe = 15;
// const int k_limit = 15;
// const int fine_k = 450;

const char* dataset_learn = "./sift/sift10k/siftsmall_learn.fvecs";
const char* dataset_base = "./sift/sift10k/siftsmall_base.fvecs";
const char* dataset_query = "./sift/sift10k/siftsmall_query.fvecs";
const char* dataset_gt = "./sift/sift10k/siftsmall_groundtruth.ivecs";

const char* index_key = "IVF256,PQ8";
const size_t nprobe = 10;
const int k_limit = 15;
const int fine_k = 300;

void sort_floats_with_labels(float* arr, faiss::idx_t* labels, size_t n) {
    std::vector<std::pair<float, faiss::idx_t>> paired(n);

    for (size_t i = 0; i < n; ++i) {
        paired[i] = {arr[i], labels[i]};
    }

    std::sort(paired.begin(), paired.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    for (size_t i = 0; i < n; ++i) {
        arr[i] = paired[i].first;
        labels[i] = paired[i].second;
    }
}

// function adapted from demo_sift1M from faiss demos.
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

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int add_dist(size_t index, size_t nprobe, float dist, float* temp_nprobe_store, faiss::idx_t* temp_nprobe_store_indexes) {
    int i;
    // Find position to insert
    for (i = 0; i < nprobe; i++) {
        if (dist < temp_nprobe_store[i]) break;
    }
    if (i == nprobe) return 0; 

    if (nprobe - i - 1 > 0) {
        memmove(&temp_nprobe_store[i + 1], &temp_nprobe_store[i], (nprobe - i - 1) * sizeof(float));
        memmove(&temp_nprobe_store_indexes[i + 1], &temp_nprobe_store_indexes[i], (nprobe - i - 1) * sizeof(faiss::idx_t));
    }

    temp_nprobe_store[i] = dist;
    temp_nprobe_store_indexes[i] = index;
    return 1;
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

  /**************************
    * READING ALL THE REQUIRED DATA 
    * AND SETTING UP THE INDEX
  ***************************/
 
  float *x_learn = fvecs_read(dataset_learn, &d, &n_learn);
  printf("n_learn : %ld, d : %ld\n", n_learn, d);
  
  float *x_base = fvecs_read(dataset_base, &d, &n_base);
  printf("n_base : %ld\n", n_base);

  int *gt_int = (int *) fvecs_read(dataset_gt , &k, &n_gt);
  float *x_query = fvecs_read(dataset_query , &d, &n_query);
  printf("n_query : %ld\n", n_query);
  printf("k : %ld\n", k);

  // create the index
  printf("Creating index : %s\n", index_key);
  index = faiss::index_factory(d, index_key);

  /**************************
    * TRAINING AND ADDING VECTORS TO THE DATABASE 
    ************************/


  if (!(std::filesystem::exists("ivf256_pq8_index.faiss"))) {
    printf("No cached data\n");
    printf("[%.3f s] Starting Training\n", elapsed()-t0);
    index->train(n_learn, x_learn);

    delete[] x_learn;

    printf("[%.3f s] Adding vectors into database\n", elapsed()-t0);
    index->add(n_base, x_base);

    delete[] x_base;

    printf("[%.3f s] Finished Adding vectors to the database\n", elapsed()-t0);

    faiss::write_index(index, "ivf256_pq8_index.faiss");

    printf("Data cached successfully.\n");

  } else {
    printf("Loading cached data\n");
    index = faiss::read_index("ivf256_pq8_index.faiss"); 
    printf("Cached data read correctly\n");
  }

  auto *ivf = dynamic_cast<faiss::IndexIVF*>(index);
  if (ivf){
    // do nothing.
  } else throw std::runtime_error("Error casting index to IVF index."); 

  printf("nlist: %ld\n", ivf->nlist);

  // ivf->nprobe = 10;
  // size_t nprobe = ivf->nprobe;
  //
  size_t max_list_size = 0;
  for (size_t i = 0; i < ivf->nlist; ++i) {
      size_t sz = ivf->invlists->list_size(i);
        if (sz > max_list_size) max_list_size = sz;
  }
  
  ivf->nprobe = nprobe;
  faiss::idx_t* labels = new faiss::idx_t[n_query * nprobe * max_list_size];
  float* distances = new float[n_query * nprobe * max_list_size];
  faiss::idx_t* centroid_indexes = new faiss::idx_t[n_query*nprobe];

  /************************************
    * SETTING UP CENTROIDS AND CENTROID SEARCH 
    **********************************/

  // setting centroid_indexes.
  printf("[%.3f s] Computing centroid_indexes\n", elapsed()-t0);
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
    for (int j = 0; j<ivf->nlist; j++) {
      float dist = compute_distance(&centroid_values[j*d], &x_query[ith_query*d]);
      add_dist(j, nprobe, dist, temp_nprobe_store, temp_nprobe_store_indexes);
    }

    for (int i = 0; i<nprobe; i++){
      assert(temp_nprobe_store_indexes[i] < ivf->nlist);
      centroid_indexes[ith_query * nprobe + i] = temp_nprobe_store_indexes[i];
    }
  }

  size_t* list_sizes_per_query = new size_t[n_query];

  /*************************************
    * PERFORM SEARCH AFTER CENTROID SEARCH IS COMPLETED
    ***********************************/


  printf("[%.3f s] Starting Search\n", elapsed()-t0);

  double start = elapsed();
  // ivf->search(n_query, x_query, k, distances, labels);
  ivf->search_encrypted(n_query, x_query,centroid_indexes, distances, labels, list_sizes_per_query);
  double searchtime = elapsed() - start;

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

  /***********************************
    * SORTING COARSE DISTANCES 
    *********************************/

  printf("[%.3f s] Starting sorting\n", elapsed()-t0);

  faiss::idx_t to_add = 0;
  for (int i = 0; i<n_query; i++) {
    sort_floats_with_labels(&distances[to_add], &labels[to_add], list_sizes_per_query[i]);
    to_add += list_sizes_per_query[i];
  }


  printf("[%.3f s] Time taken to sort the data\n", elapsed() - t0);

  delete[] distances;


  /************************************** 
    * RECALL COMPUTATION 
    ************************************/

  // which percentage of the true top-k nearest neighbours were retrieved
  printf("[%.3f s] Starting recall computation\n", elapsed()-t0);

  faiss::idx_t* gt = new faiss::idx_t[k * n_query];
  for (int i = 0; i < k * n_query; i++) {
      gt[i] = gt_int[i];
  }

  to_add = 0;
  int* n_k = new int[k_limit];
  for (int i = 0; i<k_limit; i++){
    n_k[i] = 0;
  }

  int total_retrieved = 0;
  int total_gt = k * n_query;

  for (int i = 0; i < n_query; i++) {
      faiss::idx_t query_list_size = list_sizes_per_query[i];
      for (int j = 0; j < k; j++) {
          int gt_nn = gt[i * k + j];
          for (int l = 0; l < query_list_size; l++) {
              if (labels[to_add + l] == gt_nn) {
                  total_retrieved++;
                  break;  // Count each gt_nn only once
              }
          }
      }
      to_add += query_list_size;
  }

  float relative_recall = float(total_retrieved) / total_gt;
  printf("Relative Recall@%ld = %.4f\n", k, relative_recall);

  // faiss::idx_t* gt = new faiss::idx_t[k * n_query];
  // for (int i = 0; i < k * n_query; i++) {
  //     gt[i] = gt_int[i];
  // }
  //
  to_add = 0;
  int n_1 = 0, n_10 = 0, n_100 = 0;
  float mrr_10 = 0, mrr_100 = 0;
  for (int i = 0; i < n_query; i++) {
      // for each query find the nearest neighbour.
      int gt_nn = gt[i * k];
      for (int j = 0; j < list_sizes_per_query[i]; j++) {
          // check if the query is within 
          // 1, 10 or 100.
          if (labels[to_add + j] == gt_nn) {
              if (j < 1)
                  n_1++;
              if (j < 10){
                  n_10++;
                  mrr_10 += float(1)/float(j+1);
              }
              if (j < 100) {
                  n_100++;
                  mrr_100 += float(1)/float(j+1);
              }
              break;
          }
      }
      to_add += list_sizes_per_query[i];
  }


  printf("R@1 = %.4f\n", n_1 / float(n_query));
  printf("R@10 = %.4f\n", n_10 / float(n_query));
  printf("R@100 = %.4f\n", n_100 / float(n_query));

  printf("MRR@10 = %.4f\n", mrr_10/float(n_query));
  printf("MRR@100 = %.4f\n", mrr_100/float(n_query));

  printf("[%.3f s] finished recall computation\n", elapsed()-t0);

  /**********************************************
    * FINE SEARCH 
    ********************************************/

  printf("[%.3f s] Starting fine search\n", elapsed()-t0);

  x_base = fvecs_read(dataset_base, &d, &n_base);
  start = elapsed();
  faiss::idx_t* closest_labels = new faiss::idx_t[n_query];

  to_add = 0;

  for (int i = 0; i < n_query; i++) {
      float* query = &x_query[i * d];

      float min_dist = 1e10;
      faiss::idx_t best_label = -1;

      // this k will be 100 if there are 100 vectors to fine search through 
      // for the query, if there are less than 100 it is set to the maximum 
      // vectors assigned to the query.
      k = (list_sizes_per_query[i] > fine_k) ? fine_k :list_sizes_per_query[i];
      // k = list_sizes_per_query[i];

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
  int k_final = 100;
  for (int i = 0; i < n_query; i++){
    if (closest_labels[i] == gt_int[i*k_final]){
      n_1++;
    }
  }

  double finesearchtime = elapsed() - start; 
  printf("finesearchtime : %f\n", finesearchtime);
  printf("R@1 IF WE DO FINE SEARCH THROUGH TOP %d (SELECTED ACROSS EACH QUERY): %f\n", fine_k, n_1/float(n_query));

  printf("[%.3f s] total number of vectors : %ld\n",elapsed()-t0,  offset);
  printf("Total search time : %f\n", searchtime + finesearchtime);
}
