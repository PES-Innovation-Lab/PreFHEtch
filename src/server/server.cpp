#include <memory>

#include "server_lib.h"
#include "server_utils.h"

int main() {
    std::shared_ptr<Server> server = Server::getInstance();
    server->init_index();
    server->run_webserver();
    return 0;
}
