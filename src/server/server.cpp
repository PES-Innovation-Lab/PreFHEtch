#include <memory>

#include "server_lib.h"
#include "server_utils.h"

int main() {
    init_logger();

    std::shared_ptr<Server> server = Server::getInstance();
    server->init_index("IVF128,Flat");
    server->run_webserver();
    return 0;
}
