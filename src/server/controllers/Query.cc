#include <memory>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "Query.h"
#include "server_lib.h"

void Query::ping(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /ping");
    nlohmann::json ret;
    ret["ping"] = "pong";

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(ret.dump());

    callback(resp);
}

void Query::query(
    const HttpRequestPtr &req,
    std::function<void(const HttpResponsePtr &)> &&callback) const {
    SPDLOG_INFO("Received request on /query");

    std::shared_ptr<Server> srvr = Server::getInstance();

    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> centroids;
    srvr->retrieve_centroids(centroids);
    const nlohmann::json centroids_json = centroids;

    const HttpResponsePtr resp = HttpResponse::newHttpResponse();
    resp->setContentTypeString("application/json");
    resp->setBody(centroids_json.dump());

    callback(resp);
}
