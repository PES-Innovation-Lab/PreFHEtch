#include "spdlog/spdlog.h"

#include "Query.h"
#include "server_lib.h"

void Query::ping(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /ping");
    Json::Value ret;
    ret["ping"] = "pong";

    auto resp = HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /query");

    std::vector<float> centroids = get_centroids();
    Json::Value ret = get_centroids_json(centroids);

    auto resp = HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}