#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class Query : public drogon::HttpController<Query> {
  public:
    METHOD_LIST_BEGIN

    ADD_METHOD_TO(Query::ping, "/ping", Get);

    // Endpoint to start search
    // Accepts: None
    // Returns: centroids (quantisation parameters to be implemented)
    ADD_METHOD_TO(Query::query, "/query", Get);

    // Endpoint to perform a coarse search
    // Accepts: Nearest centroid indexes (std::vector<faiss_idx_t>), coarse query
    // vector (std::array<float>) Returns: Coarse distance scores
    // (std::vector<float>)
    ADD_METHOD_TO(Query::coarse_search, "/coarsesearch", Post);
    //
    // // Endpoint to perform a precise search
    // // Accepts: Nearest coarse vector - indexes (std::vector<faiss_idx_t>),
    // precise query vector (std::array<float>)
    // // Returns: Precise distance scores (std::vector<float>)
    // ADD_METHOD_TO(Query::precise_search, "/precise-search", Post);
    //
    // // Endpoint to retrieve vectors
    // // Accepts: Nearest precise vector - indexes (std::vector<faiss_idx_t>)
    // // Returns: Query results (std::vector<float>)
    // ADD_METHOD_TO(Query::precise_vector_pir, "/precise-vector-pir", Post);

    METHOD_LIST_END

    void ping(const HttpRequestPtr &req,
              std::function<void(const HttpResponsePtr &)> &&callback) const;

    void query(const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback) const;

    void coarse_search(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;

    // void precise_search(const HttpRequestPtr &req,
    //            std::function<void(const HttpResponsePtr &)> &&callback)
    //            const;
    //
    // void precise_vector_pir(const HttpRequestPtr &req,
    //            std::function<void(const HttpResponsePtr &)> &&callback)
    //            const;
};
