#!/usr/bin/python3

import subprocess
import os
import sys
from datetime import datetime
import configparser
import time
import threading
import queue

SERVER_EXECUTABLE = "build/PreFHEtch_server"
CLIENT_EXECUTABLE = "build/PreFHEtch_client"
SERVER_READY_MESSAGE = "Server listening on"


def capture_server_output(process, output_queue):
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                output_queue.put(line.strip())
    except Exception as e:
        output_queue.put(f"Error capturing server output: {e}")


def force_kill_process(process, process_name):
    if process and process.poll() is None:
        print(f"Force killing {process_name}...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print(f"{process_name} forcefully terminated.")


def run_server_client_flow(run_name, config_params, client_output_file):
    server_process = None
    server_output_queue = queue.Queue()
    server_output_lines = []
    test_failed = False
    failure_reason = ""

    server_args = [
        SERVER_EXECUTABLE,
        "--nlist",
        config_params["nlist"],
        "--sub-quantizers",
        config_params["sub-quantizers"],
        "--sub-quantizers-size",
        config_params["sub-quantizers-size"],
        "--poly-modulus",
        config_params["poly-modulus"],
        "--plaintext-modulus",
        config_params["plaintext-modulus"],
    ]

    client_args = [
        CLIENT_EXECUTABLE,
        "--nq",
        config_params["nq"],
        "--nprobe",
        config_params["nprobe"],
        "--coarse-probe",
        config_params["coarse-probe"],
        "--k",
        config_params["k"],
    ]

    if config_params.getboolean("single-phase", fallback=False):
        client_args.append("--single-phase")

    try:
        print(f"Starting server: {' '.join(server_args)}")
        try:
            server_process = subprocess.Popen(
                server_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            test_failed = True
            failure_reason = f"Server executable not found: {SERVER_EXECUTABLE}"
            raise

        server_thread = threading.Thread(
            target=capture_server_output, args=(server_process, server_output_queue)
        )
        server_thread.daemon = True
        server_thread.start()

        print(f"Waiting for the server to log: '{SERVER_READY_MESSAGE}'...")

        server_ready = False

        while True:
            try:
                line = server_output_queue.get(timeout=1)
                server_output_lines.append(line)
                print(f"[SERVER] {line}")

                if SERVER_READY_MESSAGE in line:
                    print("Server ready message received!")
                    server_ready = True
                    break
            except queue.Empty:
                if server_process.poll() is not None:
                    test_failed = True
                    failure_reason = (
                        "Server process terminated unexpectedly during startup"
                    )
                    print("Server process has terminated unexpectedly.")
                    break
                continue

        if not server_ready:
            test_failed = True
            if not failure_reason:
                failure_reason = "Server failed to start"
            print(
                "ERROR: Server process exited before becoming ready. Aborting this run."
            )
            force_kill_process(server_process, "server")
            raise Exception(failure_reason)

        time.sleep(0.5)

        print(f"\nRunning client for '{run_name}' with args: {' '.join(client_args)}")

        try:
            client_result = subprocess.run(client_args, capture_output=True, text=True)
        except FileNotFoundError:
            test_failed = True
            failure_reason = f"Client executable not found: {CLIENT_EXECUTABLE}"
            raise

        if client_result.returncode != 0:
            test_failed = True
            failure_reason = (
                f"Client process failed with return code: {client_result.returncode}"
            )
            print(
                f"Client process completed with return code: {client_result.returncode}"
            )
        else:
            print("Client process completed successfully.")

        time.sleep(1)
        while not server_output_queue.empty():
            try:
                line = server_output_queue.get_nowait()
                server_output_lines.append(line)
            except queue.Empty:
                break

    except FileNotFoundError as e:
        test_failed = True
        if not failure_reason:
            failure_reason = f"Executable not found: {e}"
        print("ERROR: Could not find an executable. Make sure paths are correct.")
        print(e)
        client_result = None
    except Exception as e:
        test_failed = True
        if not failure_reason:
            failure_reason = f"Unexpected error: {e}"
        print(f"An unexpected error occurred during run '{run_name}': {e}")
        client_result = None
    finally:
        force_kill_process(server_process, "server")

    log_test_results(
        run_name,
        config_params,
        server_args,
        client_args,
        server_output_lines,
        client_result,
        test_failed,
        failure_reason,
        client_output_file,
    )

    if test_failed:
        print(f"Test run '{run_name}' FAILED: {failure_reason}")
        return False
    else:
        print(f"Test run '{run_name}' COMPLETED successfully")
        return True


def log_test_results(
    run_name,
    config_params,
    server_args,
    client_args,
    server_output_lines,
    client_result,
    test_failed,
    failure_reason,
    client_output_file,
):
    print(f"Logging results to {client_output_file}")

    with open(client_output_file, "a") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"TEST RUN: {run_name}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Status: {'FAILED' if test_failed else 'SUCCESS'}\n")
        if test_failed:
            f.write(f"Failure Reason: {failure_reason}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write("--- Configuration Parameters ---\n")
        for key, value in config_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("--- Boolean Options Status ---\n")
        single_phase_status = config_params.getboolean("single-phase", fallback=False)
        f.write(f"single-phase: {'ENABLED' if single_phase_status else 'DISABLED'}\n")
        f.write("\n")

        f.write("--- Server Arguments ---\n")
        f.write(f"{' '.join(server_args)}\n\n")

        f.write("--- Client Arguments ---\n")
        f.write(f"{' '.join(client_args)}\n\n")

        f.write("--- Server Output ---\n")
        if server_output_lines:
            for line in server_output_lines:
                f.write(f"{line}\n")
        else:
            f.write("No server output captured\n")
        f.write("\n")

        if client_result is not None:
            f.write("--- Client Standard Output ---\n")
            f.write(
                client_result.stdout if client_result.stdout else "No client stdout\n"
            )
            f.write("\n--- Client Standard Error ---\n")
            f.write(
                client_result.stderr if client_result.stderr else "No client stderr\n"
            )
            f.write(f"\n--- Client Return Code ---\n")
            f.write(f"{client_result.returncode}\n")
        else:
            f.write("--- Client Output ---\n")
            f.write("Client process did not complete successfully\n")

        if test_failed:
            f.write(f"\n--- FAILURE DETAILS ---\n")
            f.write(f"Reason: {failure_reason}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(f"END OF TEST RUN: {run_name}\n")
        f.write(f"{'=' * 80}\n\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file_path>")
        print("Example: python script.py config.ini")
        print("Example: python script.py /path/to/my_config.ini")
        sys.exit(1)

    config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    client_output_file = datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.log")

    with open(client_output_file, "w") as f:
        f.write(f"PreFHEtch Test Run Log\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Configuration file: {config_file}\n")
        f.write(f"{'=' * 80}\n\n")

    print(f"Log file created: {client_output_file}")
    print(f"Using configuration file: {config_file}")

    total_tests = len(config.sections())
    successful_tests = 0
    failed_tests = 0

    print(f"\nStarting {total_tests} test runs...")

    for section in config.sections():
        print(f"\n{'=' * 20} Starting Test Run: {section} {'=' * 20}")

        if run_server_client_flow(section, config[section], client_output_file):
            successful_tests += 1
        else:
            failed_tests += 1

        print(f"{'=' * 20} Finished Test Run: {section} {'=' * 20}")

    print(f"\nALL TEST RUNS COMPLETED")
    print(f"Summary:")
    print(f"   Successful: {successful_tests}/{total_tests}")
    print(f"   Failed: {failed_tests}/{total_tests}")
    print(f"   Results saved to: {client_output_file}")

    with open(client_output_file, "a") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"BENCHMARK SUMMARY\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {(successful_tests / total_tests) * 100:.1f}%\n")
        f.write(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
