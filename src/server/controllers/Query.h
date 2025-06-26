#pragma once

#include <drogon/HttpController.h>
#include <functional>

using namespace drogon;

class Query : public drogon::HttpController<Query> {
  public:
    METHOD_LIST_BEGIN

    ADD_METHOD_TO(Query::ping, "/ping", Get);

    METHOD_LIST_END

    void ping(const HttpRequestPtr &req,
              std::function<void(const HttpResponsePtr &)> &&callback) const;
};
