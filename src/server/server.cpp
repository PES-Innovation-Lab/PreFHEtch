#include <memory>

#include "client_server_utils.h"
#include "server_lib.h"
#include "server_utils.h"

int main() {
    std::shared_ptr<Server> server = Server::getInstance();

    Timer init_index_timer;
    init_index_timer.StartTimer();
    server->init_index();
    init_index_timer.StopTimer();
    SPDLOG_INFO("Initialized index, time(microseconds) = {}",
                init_index_timer.getDurationMicroseconds());

    Server::run_webserver();
    return 0;
}
