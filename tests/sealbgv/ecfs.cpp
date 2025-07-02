#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cassert>
#include <sys/stat.h>
#include <sys/time.h>
#include <algorithm>
#include <random>
using namespace std;
using namespace seal;

const int SCALE_FACTOR = 100;
const size_t BATCH_SIZE = 64;
const int QUANTIZATION_BITS = 8;
const size_t TOP_K = 10;

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

uint64_t quantize_float(float val) {
    int quantized = static_cast<int>(val * SCALE_FACTOR);
    return static_cast<uint64_t>(quantized + (1 << (QUANTIZATION_BITS + 8)));
}

vector<vector<uint64_t>> batch_database(float* db, size_t n, size_t d, size_t slot_count) {
    size_t num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
    vector<vector<uint64_t>> batched_db(num_batches);
    
    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        batched_db[batch_idx].resize(slot_count, 0ULL);
        
        for (size_t i = 0; i < BATCH_SIZE && batch_idx * BATCH_SIZE + i < n; i++) {
            size_t db_idx = batch_idx * BATCH_SIZE + i;
            for (size_t j = 0; j < d && i * d + j < slot_count; j++) {
                batched_db[batch_idx][i * d + j] = quantize_float(db[db_idx * d + j]);
            }
        }
    }
    
    return batched_db;
}

vector<uint64_t> replicate_query(float* query, size_t d, size_t slot_count) {
    vector<uint64_t> replicated_query(slot_count, 0ULL);
    
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        for (size_t j = 0; j < d && i * d + j < slot_count; j++) {
            replicated_query[i * d + j] = quantize_float(query[j]);
        }
    }
    
    return replicated_query;
}

vector<uint64_t> extract_batch_scores(const vector<uint64_t>& result, size_t d) {
    vector<uint64_t> scores;
    
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        if (i * d < result.size()) {
            scores.push_back(result[i * d]);
        }
    }
    
    return scores;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double t0 = elapsed();
    
    size_t d, n, nq;
    float* db = fvecs_read("/home/cheese/Desktop/sealbgv/siftsmall/siftsmall_base.fvecs", &d, &n);
    float* queries = fvecs_read("/home/cheese/Desktop/sealbgv/siftsmall/siftsmall_query.fvecs", &d, &nq);
    float* query = queries + 0 * d;

    cout << "Dataset: " << n << " vectors of dimension " << d << endl;
    cout << "Batch size: " << BATCH_SIZE << " vectors per ciphertext" << endl;

    EncryptionParameters parms(scheme_type::bgv);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    
    SEALContext context(parms);
    cout << "BGV parameters: poly_degree=" << poly_modulus_degree << endl;
    
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);
    
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    BatchEncoder batch_encoder(context);
    
    size_t slot_count = batch_encoder.slot_count();
    cout << "Slot count: " << slot_count << endl;
    
    if (d * BATCH_SIZE > slot_count) {
        cout << "Warning: Batch size too large for slot count. Reducing batch size." << endl;
    }

    cout << "Preprocessing database into batches..." << endl;
    double preprocess_start = elapsed();
    
    vector<vector<uint64_t>> batched_db = batch_database(db, n, d, slot_count);
    
    vector<Plaintext> encoded_batches(batched_db.size());
    for (size_t i = 0; i < batched_db.size(); i++) {
        batch_encoder.encode(batched_db[i], encoded_batches[i]);
    }
    
    double preprocess_time = elapsed() - preprocess_start;
    cout << "Preprocessing completed in " << preprocess_time << " seconds" << endl;
    cout << "Created " << batched_db.size() << " batches" << endl;

    vector<uint64_t> replicated_query = replicate_query(query, d, slot_count);
    Plaintext plain_query;
    batch_encoder.encode(replicated_query, plain_query);
    Ciphertext encrypted_query;
    encryptor.encrypt(plain_query, encrypted_query);

    cout << "Starting batched search..." << endl;
    double search_start = elapsed();
    
    vector<pair<uint64_t, size_t>> all_scores;
    
    for (size_t batch_idx = 0; batch_idx < encoded_batches.size(); batch_idx++) {
        Ciphertext product;
        evaluator.multiply_plain(encrypted_query, encoded_batches[batch_idx], product);
        evaluator.relinearize_inplace(product, relin_keys);
        
        Ciphertext result = product;
        for (size_t step = 1; step < d; step *= 2) {
            Ciphertext rotated;
            evaluator.rotate_rows(result, step, galois_keys, rotated);
            evaluator.add_inplace(result, rotated);
        }
        
        Plaintext plain_result;
        decryptor.decrypt(result, plain_result);
        vector<uint64_t> batch_result;
        batch_encoder.decode(plain_result, batch_result);
        
        vector<uint64_t> scores = extract_batch_scores(batch_result, d);
        
        for (size_t i = 0; i < scores.size() && batch_idx * BATCH_SIZE + i < n; i++) {
            size_t original_idx = batch_idx * BATCH_SIZE + i;
            all_scores.push_back({scores[i], original_idx});
        }
        
        if (batch_idx % 10 == 0) {
            cout << "Processed batch " << batch_idx << "/" << batched_db.size() << endl;
        }
    }
    
    double search_time = elapsed() - search_start;
    
    sort(all_scores.begin(), all_scores.end(), greater<pair<uint64_t, size_t>>());
    
    cout << "\nTop " << min(TOP_K, (size_t)all_scores.size()) << " highest scoring vectors:" << endl;
    for (size_t i = 0; i < min(TOP_K, (size_t)all_scores.size()); i++) {
        uint64_t score = all_scores[i].first;
        size_t idx = all_scores[i].second;
        
        double approx_score = static_cast<double>(score) / (SCALE_FACTOR * SCALE_FACTOR);
        cout << "Rank " << i+1 << ": Index " << idx 
             << ", Score ≈ " << approx_score << endl;
    }
    
    cout << "\nTiming breakdown:" << endl;
    cout << "Database preprocessing: " << preprocess_time << " seconds" << endl;
    cout << "Batched search time: " << search_time << " seconds" << endl;
    cout << "Vectors per second: " << (double)n / search_time << endl;
    printf("Total time: [%.3f s]\n", elapsed() - t0);

    return 0;
}
