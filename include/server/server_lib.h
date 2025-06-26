#pragma once

#include <vector>

#include "drogon/drogon.h"
#include "json/json.h"

void init_logger();
void run_server();

std::vector<float> get_centroids();
Json::Value get_centroids_json(const std::vector<float>& centroids);
