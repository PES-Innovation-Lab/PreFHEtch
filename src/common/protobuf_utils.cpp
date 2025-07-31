#include "protobuf_utils.h"
#include <seal/seal.h>

namespace prefhetch_utils {

void vector_to_repeated(const std::vector<float>& vec, google::protobuf::RepeatedField<float>* repeated) {
    repeated->Clear();
    repeated->Reserve(vec.size());
    for (const auto& val : vec) {
        repeated->Add(val);
    }
}

void repeated_to_vector(const google::protobuf::RepeatedField<float>& repeated, std::vector<float>& vec) {
    vec.clear();
    vec.reserve(repeated.size());
    for (int i = 0; i < repeated.size(); ++i) {
        vec.push_back(repeated.Get(i));
    }
}

void vector_to_repeated(const std::vector<uint64_t>& vec, google::protobuf::RepeatedField<uint64_t>* repeated) {
    repeated->Clear();
    repeated->Reserve(vec.size());
    for (const auto& val : vec) {
        repeated->Add(val);
    }
}

void repeated_to_vector(const google::protobuf::RepeatedField<uint64_t>& repeated, std::vector<uint64_t>& vec) {
    vec.clear();
    vec.reserve(repeated.size());
    for (int i = 0; i < repeated.size(); ++i) {
        vec.push_back(repeated.Get(i));
    }
}

void vector_to_repeated(const std::vector<seal::seal_byte>& vec, google::protobuf::RepeatedField<uint8_t>* repeated) {
    repeated->Clear();
    repeated->Reserve(vec.size());
    for (const auto& val : vec) {
        repeated->Add(static_cast<uint8_t>(val));
    }
}

void repeated_to_vector(const google::protobuf::RepeatedField<uint8_t>& repeated, std::vector<seal::seal_byte>& vec) {
    vec.clear();
    vec.reserve(repeated.size());
    for (int i = 0; i < repeated.size(); ++i) {
        vec.push_back(static_cast<seal::seal_byte>(repeated.Get(i)));
    }
}

} // namespace prefhetch_utils 