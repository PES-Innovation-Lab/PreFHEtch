#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class Query : public drogon::HttpController<Query> {
  public:
    METHOD_LIST_BEGIN

    ADD_METHOD_TO(Query::query, "/query", Get);

    ADD_METHOD_TO(Query::coarseSearch, "/coarsesearch", Post);

    ADD_METHOD_TO(Query::precise_search, "/precisesearch", Post);

    ADD_METHOD_TO(Query::precise_vector_pir, "/precise-vector-pir", Post);

    ADD_METHOD_TO(Query::single_phase_search, "/single-phase-search", Post);

    METHOD_LIST_END

    void query(const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback) const;

    void coarseSearch(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback);

    void precise_search(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;

    void precise_vector_pir(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;

    void single_phase_search(
        const HttpRequestPtr &req,
        std::function<void(const HttpResponsePtr &)> &&callback) const;
};
