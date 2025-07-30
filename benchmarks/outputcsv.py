#!/usr/bin/python3

import re
import csv

def parse_log_to_csv(log_file_path, output_csv_path):
    with open(log_file_path, 'r') as f:
        log_data = f.read()

    log_entries_raw = log_data.split('--- Log entry from:')
    
    all_rows_data = []

    for entry_text in log_entries_raw:
        if not entry_text.strip():
            continue

        timestamps = [
            int(time_val)
            for time_val in re.findall(r'time\(microseconds\) = (\d+)', entry_text)
        ]
        
        if len(timestamps) == 7:
            total_time_microseconds = sum(timestamps)
            total_time_milliseconds = total_time_microseconds / 1000

            row_data = [t / 1000 for t in timestamps]
            row_data.append(total_time_milliseconds)
            all_rows_data.append(row_data)
        else:
            print(f"Skipping an incomplete log entry. Found {len(timestamps)} timestamps, expected 7.")

    if not all_rows_data:
        print("No complete log entries found to parse. Please check the log file.")
        return

    header = [
        'Get Query Vectors (ms)',
        'Fetch Centroids (ms)',
        'Compute Nearest Centroids (ms)',
        'Compute Encrypted Subvector (ms)',
        'Receive Encrypted Coarse Distances (ms)',
        'Deserialize and Decrypt Coarse Distances (ms)',
        'Compute Nearest Coarse Vectors (ms)',
        'Total Time (ms)'
    ]

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(all_rows_data)

    print(f"Successfully created CSV file at: {output_csv_path} with {len(all_rows_data)} entries.")

log_file = 'client_output.log'
output_csv = 'parsed_log_output.csv'
parse_log_to_csv(log_file, output_csv)
