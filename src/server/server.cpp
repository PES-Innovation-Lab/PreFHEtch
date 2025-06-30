#include "server_lib.h"

int main() {
    init_logger();
    init_index();
    run_webserver();
    return 0;
}
