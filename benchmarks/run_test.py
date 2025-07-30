#!/usr/bin/python3

import subprocess
import os
from datetime import datetime
import configparser
import time

SERVER_EXECUTABLE = './build/PreFHEtch_server'
CLIENT_EXECUTABLE = './build/PreFHEtch_client'
CONFIG_FILE = 'config.ini'
SERVER_READY_MESSAGE = 'Server listening on'
CLIENT_OUTPUT_FILE = 'client_output.log'

def run_server_client_flow(run_name, config_params):
    """
    Starts the server, waits for it to be ready, runs the client with specific
    arguments, and then terminates the server.
    """
    server_process = None

    server_args = [
        SERVER_EXECUTABLE,
        '--nlist', '256',
        '--subquantizer', '32',
        '--subquantizer-size', '8'
    ]

    client_args = [
        CLIENT_EXECUTABLE,
        '--nq', config_params['nq'],
        '--nprobe', config_params['nprobe'],
        '--coarse-probe', config_params['coarse-probe'],
        '--k', config_params['k']
    ]

    try:
        print(f"Starting server: {' '.join(server_args)}")
        server_process = subprocess.Popen(
            server_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        print(f"Waiting for the server to log: '{SERVER_READY_MESSAGE}'...")

        server_ready = False
        for line in iter(server_process.stdout.readline, ''):
            print(f"[SERVER] {line.strip()}")
            if SERVER_READY_MESSAGE in line:
                print("Server ready message received!")
                server_ready = True
                break

        if not server_ready:
            print("ERROR: Server process exited before becoming ready. Aborting this run.")
            if server_process.poll() is None:
                    server_process.terminate()
            return

        time.sleep(0.5)

        print(f"\nRunning client for '{run_name}' with args: {' '.join(client_args)}")
        client_result = subprocess.run(
            client_args,
            capture_output=True,
            text=True,
            timeout=300
        )

        print(f"Client process finished. Appending output to {CLIENT_OUTPUT_FILE}")
        with open(CLIENT_OUTPUT_FILE, 'a') as f:
            f.write(f"\n--- Log for run '{run_name}' at: {datetime.now()} ---\n")
            f.write(f"Client arguments: {' '.join(client_args)}\n")
            f.write("--- Client Standard Output ---\n")
            f.write(client_result.stdout)
            f.write("\n--- Client Standard Error ---\n")
            f.write(client_result.stderr)
            f.write("\n--- End of Log Entry ---\n")

        print("Client output successfully appended.")

    except FileNotFoundError as e:
        print(f"ERROR: Could not find an executable. Make sure paths are correct.")
        print(e)
    except subprocess.TimeoutExpired:
        print(f"ERROR: Client process for run '{run_name}' timed out.")
    except Exception as e:
        print(f"An unexpected error occurred during run '{run_name}': {e}")
    finally:
        if server_process:
            print("\nShutting down server...")
            server_process.terminate()
            server_process.wait()
            print("Server has been terminated.")

def main():
    """
    Main function to read configurations and execute the runs.
    """
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if os.path.exists(CLIENT_OUTPUT_FILE):
        os.remove(CLIENT_OUTPUT_FILE)
        print(f"Cleared previous log file: {CLIENT_OUTPUT_FILE}")

    for section in config.sections():
        print(f"\n{'='*20} Starting Test Run: {section} {'='*20}")
        run_params = config[section]
        run_server_client_flow(section, run_params)
        print(f"{'='*20} Finished Test Run: {section} {'='*20}")

if __name__ == "__main__":
    main()
