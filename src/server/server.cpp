#include <memory>

#include "client_server_utils.h"
#include "server_lib.h"

int main() {
    if (BFV_SCALING_FACTOR != 1) {
        SPDLOG_ERROR("BFV_SCALING_FACTOR = 1");
        throw std::runtime_error("BFV_SCALING_FACTOR != 1");
    }

    std::shared_ptr<Server> server = Server::getInstance();

    Timer init_index_timer;
    init_index_timer.StartTimer();
    server->init_index();
    init_index_timer.StopTimer();
    SPDLOG_INFO("Initialized index, time = {}(us)",
                init_index_timer.getDurationMicroseconds());

    Server::run_webserver();
    return 0;
}
