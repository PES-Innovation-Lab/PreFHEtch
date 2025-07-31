#pragma once

#include <array>
#include <vector>
#include <seal/seal.h>
#include "prefhetch.pb.h"

namespace prefhetch_utils {

// Helper functions to convert between std::array and protobuf repeated fields

template<size_t N>
void array_to_repeated(const std::array<float, N>& arr, google::protobuf::RepeatedField<float>* repeated) {
    repeated->Clear();
    repeated->Reserve(N);
    for (const auto& val : arr) {
        repeated->Add(val);
    }
}

template<size_t N>
void repeated_to_array(const google::protobuf::RepeatedField<float>& repeated, std::array<float, N>& arr) {
    if (repeated.size() != N) {
        throw std::runtime_error("Protobuf repeated field size doesn't match array size");
    }
    for (size_t i = 0; i < N; ++i) {
        arr[i] = repeated.Get(i);
    }
}

template<size_t N>
void array_to_repeated(const std::array<uint64_t, N>& arr, google::protobuf::RepeatedField<uint64_t>* repeated) {
    repeated->Clear();
    repeated->Reserve(N);
    for (const auto& val : arr) {
        repeated->Add(val);
    }
}

template<size_t N>
void repeated_to_array(const google::protobuf::RepeatedField<uint64_t>& repeated, std::array<uint64_t, N>& arr) {
    if (repeated.size() != N) {
        throw std::runtime_error("Protobuf repeated field size doesn't match array size");
    }
    for (size_t i = 0; i < N; ++i) {
        arr[i] = repeated.Get(i);
    }
}

// For nested arrays (2D)
template<size_t N, size_t M>
void nested_array_to_repeated(const std::array<std::array<float, M>, N>& arr, google::protobuf::RepeatedField<float>* repeated) {
    repeated->Clear();
    repeated->Reserve(N * M);
    for (const auto& inner_arr : arr) {
        for (const auto& val : inner_arr) {
            repeated->Add(val);
        }
    }
}

template<size_t N, size_t M>
void repeated_to_nested_array(const google::protobuf::RepeatedField<float>& repeated, std::array<std::array<float, M>, N>& arr) {
    if (repeated.size() != N * M) {
        throw std::runtime_error("Protobuf repeated field size doesn't match nested array size");
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            arr[i][j] = repeated.Get(i * M + j);
        }
    }
}

// For 3D nested arrays
template<size_t N, size_t M, size_t K>
void triple_nested_array_to_repeated(const std::array<std::array<std::array<float, K>, M>, N>& arr, google::protobuf::RepeatedField<float>* repeated) {
    repeated->Clear();
    repeated->Reserve(N * M * K);
    for (const auto& outer_arr : arr) {
        for (const auto& inner_arr : outer_arr) {
            for (const auto& val : inner_arr) {
                repeated->Add(val);
            }
        }
    }
}

template<size_t N, size_t M, size_t K>
void repeated_to_triple_nested_array(const google::protobuf::RepeatedField<float>& repeated, std::array<std::array<std::array<float, K>, M>, N>& arr) {
    if (repeated.size() != N * M * K) {
        throw std::runtime_error("Protobuf repeated field size doesn't match triple nested array size");
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < K; ++k) {
                arr[i][j][k] = repeated.Get(i * M * K + j * K + k);
            }
        }
    }
}

// Vector to repeated field
void vector_to_repeated(const std::vector<float>& vec, google::protobuf::RepeatedField<float>* repeated);
void repeated_to_vector(const google::protobuf::RepeatedField<float>& repeated, std::vector<float>& vec);

void vector_to_repeated(const std::vector<uint64_t>& vec, google::protobuf::RepeatedField<uint64_t>* repeated);
void repeated_to_vector(const google::protobuf::RepeatedField<uint64_t>& repeated, std::vector<uint64_t>& vec);

void vector_to_repeated(const std::vector<seal::seal_byte>& vec, google::protobuf::RepeatedField<uint8_t>* repeated);
void repeated_to_vector(const google::protobuf::RepeatedField<uint8_t>& repeated, std::vector<seal::seal_byte>& vec);

} // namespace prefhetch_utils 