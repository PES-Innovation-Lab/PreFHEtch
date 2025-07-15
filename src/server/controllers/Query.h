#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class Query : public drogon::HttpController<Query> {
  public:
    METHOD_LIST_BEGIN

    // Endpoint to start search
    // Accepts: None
    // Returns: centroids
    ADD_METHOD_TO(Query::query, "/query", Get);

    // Endpoint to perform a coarse search
    // Accepts: Nearest centroid indexes and coarse query
    // Temporarily sending precise query
    // Returns: Coarse distance scores, coarse vector indexes and list sizes per
    // query
    ADD_METHOD_TO(Query::coarse_search, "/coarsesearch", Post);

    // Endpoint to perform a precise search
    // Accepts: Nearest coarse vector indexes and precise query
    // Returns:
    ADD_METHOD_TO(Query::precise_search, "/precisesearch", Post);

    // Endpoint to retrieve vectors
    // Accepts: Nearest precise vector indexes
    // Returns: Query results
    ADD_METHOD_TO(Query::precise_vector_pir, "/precise-vector-pir", Post);

    METHOD_LIST_END

    void query(const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback) const;

    void coarse_search(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;

    void precise_search(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;

    void precise_vector_pir(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;
};
