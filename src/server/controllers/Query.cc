#include "spdlog/spdlog.h"

#include "Query.h"

void Query::ping(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {

    SPDLOG_INFO("Received request on /ping");
    Json::Value ret;
    ret["ping"] = "pong";

    auto resp = HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}
