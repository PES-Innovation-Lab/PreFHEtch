#include <cstddef>
#include <memory>
#include <mutex>

#include <boost/program_options.hpp>

#include "client_server_utils.h"
#include "seal/encryptionparams.h"
#include "server_lib.h"

namespace po = boost::program_options;

std::shared_ptr<Server> Server::shared_instance = nullptr;
std::once_flag Server::server_initialised;
bool Server::server_initialised_bool = false;

int main(int argc, char *argv[]) {
    if (BFV_SCALING_FACTOR != 1) {
        SPDLOG_ERROR("BFV_SCALING_FACTOR = 1");
        throw std::runtime_error("BFV_SCALING_FACTOR != 1");
    }

    size_t nlist, sub_quantizers, sub_quantizers_size, poly_modulus,
        plaintext_modulus;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()("help", "help message")("nlist", po::value<size_t>(),
                                                   "number of index centroids")(
            "sub-quantizers", po::value<size_t>(),
            "number of PQ compressed subvectors")(
            "sub-quantizers-size", po::value<size_t>(),
            "number of bits to represent each PQ subvector")(
            "poly-modulus", po::value<size_t>(),
            "SEAL BFV poly-modulus degree")("plaintext-modulus",
                                            po::value<size_t>(),
                                            "SEAL BFV plaintext modulus");

        po::variables_map po_vm;
        po::store(po::parse_command_line(argc, argv, desc), po_vm);
        po::notify(po_vm);

        if (po_vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        nlist = po_vm["nlist"].as<size_t>();
        sub_quantizers = po_vm["sub-quantizers"].as<size_t>();
        sub_quantizers_size = po_vm["sub-quantizers-size"].as<size_t>();
        poly_modulus = po_vm["poly-modulus"].as<size_t>();
        plaintext_modulus = po_vm["plaintext-modulus"].as<size_t>();

    } catch (std::exception &e) {
        SPDLOG_ERROR("Error while parsing command line args = {}", e.what());
        return 1;
    }

    seal::EncryptionParameters encrypt_params(seal::scheme_type::bfv);

    encrypt_params.set_poly_modulus_degree(poly_modulus);
    encrypt_params.set_coeff_modulus(
        seal::CoeffModulus::BFVDefault(poly_modulus));
    encrypt_params.set_plain_modulus(
        seal::PlainModulus::Batching(poly_modulus, plaintext_modulus));

    seal::SEALContext seal_ctx(encrypt_params);
    SPDLOG_INFO("Encryption params = {}", seal_ctx.parameter_error_message());

    Server::intialiseServer(nlist, sub_quantizers, sub_quantizers_size,
                            poly_modulus, plaintext_modulus, encrypt_params,
                            seal_ctx);

    std::shared_ptr<Server> server = Server::getInstance();

    Timer init_index_timer;
    init_index_timer.StartTimer();
    server->init_index();
    init_index_timer.StopTimer();
    SPDLOG_INFO("Initialized index, time = {}(ms)",
                init_index_timer.getDurationMilliseconds());

    Server::run_webserver();
    return 0;
}
