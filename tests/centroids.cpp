#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);
    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    fclose(f);
    return x;
}

void fvecs_write(const char* fname, const float* data, size_t d, size_t n) {
    FILE* f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", fname);
        abort();
    }
    
    for (size_t i = 0; i < n; i++) {
        int dim = (int)d;
        fwrite(&dim, sizeof(int), 1, f);
        fwrite(data + i * d, sizeof(float), d, f);
    }
    fclose(f);
}

int main() {
    const char* input_file = "../siftsmall/siftsmall_base.fvecs";
    const char* output_file = "centroids.fvecs";
    const char* index_file = "ivf_index.faiss";
    const size_t num_centroids = 256;
    
    size_t d, n;
    std::cout << "Reading SIFT dataset..." << std::endl;
    float* data = fvecs_read(input_file, &d, &n);
    std::cout << "Read " << n << " vectors of dimension " << d << std::endl;
    
    size_t n_use = std::min(n, (size_t)10000);
    
    std::cout << "Creating FAISS IVF index with inner product metric..." << std::endl;
    
    faiss::IndexFlat* quantizer = new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT);
    
    faiss::IndexIVFFlat* index = new faiss::IndexIVFFlat(quantizer, d, num_centroids, faiss::METRIC_INNER_PRODUCT);
    
    float* normalized_data = new float[n_use * d];
    for (size_t i = 0; i < n_use; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norm = sqrt(norm);
        
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                normalized_data[i * d + j] = data[i * d + j] / norm;
            }
        } else {
            memcpy(normalized_data + i * d, data + i * d, d * sizeof(float));
        }
    }
    
    std::cout << "Training IVF index with " << num_centroids << " centroids..." << std::endl;
    index->train(n_use, normalized_data);
    
    std::cout << "Adding vectors to index..." << std::endl;
    index->add(n_use, normalized_data);
    
    faiss::IndexFlat* flat_quantizer = dynamic_cast<faiss::IndexFlat*>(index->quantizer);
    if (!flat_quantizer) {
        std::cerr << "Error: Could not cast quantizer to IndexFlat" << std::endl;
        return -1;
    }
    
    float* centroids_data = new float[num_centroids * d];
    
    for (size_t i = 0; i < num_centroids; i++) {
        flat_quantizer->reconstruct(i, centroids_data + i * d);
    }
    
    std::cout << "Saving centroids to file..." << std::endl;
    fvecs_write(output_file, centroids_data, d, num_centroids);
    
    std::cout << "Saving FAISS index..." << std::endl;
    faiss::write_index(index, index_file);
    
    std::cout << "IVF centroids saved to " << output_file << std::endl;
    std::cout << "FAISS index saved to " << index_file << std::endl;
    std::cout << "Found " << num_centroids << " IVF centroids of dimension " << d << std::endl;
    std::cout << "Using inner product metric for similarity" << std::endl;
    
    return 0;
}
