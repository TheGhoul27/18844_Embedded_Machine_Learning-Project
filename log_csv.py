# import serial
# import time
# import csv

# PORT = r"COM6"      # <-- change this to your port (e.g. "/dev/ttyACM0" on Linux/Mac)
# BAUD = 115200
# OUT_FILE = "ppg_log.csv"

# def main():
#     ser = serial.Serial(PORT, BAUD, timeout=1)
#     time.sleep(2)  # let Arduino reset

#     # Clear any junk
#     ser.reset_input_buffer()

#     # Tell Arduino to start
#     ser.write(b"START\n")
#     print("Sent START, waiting for ACK...")

#     # Wait for ACK_START
#     while True:
#         line = ser.readline().decode("utf-8", errors="ignore").strip()
#         if not line:
#             continue
#         print("ARDUINO:", line)
#         if line == "ACK_START":
#             break

#     print(f"Logging to {OUT_FILE}. Press Ctrl+C to stop.")

#     with open(OUT_FILE, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["millis", "ir", "red", "green"])

#         try:
#             while True:
#                 line = ser.readline().decode("utf-8", errors="ignore").strip()
#                 if not line:
#                     continue

#                 # Skip ACKs or other text
#                 if line.startswith("ACK") or line == "READY" or line.startswith("ERROR"):
#                     print("ARDUINO:", line)
#                     continue

#                 parts = line.split(",")
#                 if len(parts) != 4:
#                     # Not a data line, ignore
#                     continue

#                 # Optionally sanity-check that all are numbers
#                 try:
#                     millis = int(parts[0])
#                     ir     = int(parts[1])
#                     red    = int(parts[2])
#                     green  = int(parts[3])
#                 except ValueError:
#                     continue

#                 writer.writerow([millis, ir, red, green])

#         except KeyboardInterrupt:
#             print("\nStopping logging...")

#     # Tell Arduino to stop
#     ser.write(b"STOP\n")
#     time.sleep(0.2)
#     ser.close()
#     print("Serial closed.")

# if __name__ == "__main__":
#     main()


import serial
import time
import csv
import os

import matplotlib.pyplot as plt

# ===================== CONFIG ===================== #
PORT = r"COM6"      # Change if needed
BAUD = 115200
DURATION_SECONDS = 30   # How long each recording should last
OUTPUT_DIR = "."        # Folder to save CSVs
# ================================================== #


def wait_for_ack(ser, expected="ACK_START"):
    """Wait until Arduino sends a specific ACK line."""
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print("ARDUINO:", line)
        if line == expected:
            return


def record_segment(ser, duration_s, out_file):
    """Record one segment for duration_s seconds into out_file."""
    # Clear any junk before starting
    ser.reset_input_buffer()

    # Tell Arduino to start logging
    ser.write(b"START\n")
    print("Sent START, waiting for ACK_START...")
    wait_for_ack(ser, expected="ACK_START")

    print(f"Recording for {duration_s} seconds into {out_file} ...")

    start_time = time.time()
    end_time = start_time + duration_s

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["millis", "ir", "red", "green"])

        while time.time() < end_time:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Skip control / info lines
            if line.startswith("ACK") or line == "READY" or line.startswith("ERROR"):
                print("ARDUINO:", line)
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue

            try:
                millis = int(parts[0])
                ir     = int(parts[1])
                red    = int(parts[2])
                green  = int(parts[3])
            except ValueError:
                continue

            writer.writerow([millis, ir, red, green])

    # Tell Arduino to stop logging
    ser.write(b"STOP\n")
    print("Sent STOP, waiting for ACK_STOP...")
    # We don't strictly need this, but it's nice for sync
    wait_for_ack(ser, expected="ACK_STOP")
    print("Recording done.")


def plot_csv(out_file):
    """Read the CSV and plot IR, RED, GREEN vs time."""
    millis_list = []
    ir_list = []
    red_list = []
    green_list = []

    with open(out_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) != 4:
                continue
            try:
                millis = int(row[0])
                ir     = int(row[1])
                red    = int(row[2])
                green  = int(row[3])
            except ValueError:
                continue

            millis_list.append(millis)
            ir_list.append(ir)
            red_list.append(red)
            green_list.append(green)

    if not millis_list:
        print("No data to plot in", out_file)
        return

    # Convert millis to seconds, start at 0
    t0 = millis_list[0]
    t_sec = [(m - t0) / 1000.0 for m in millis_list]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_sec, ir_list)
    plt.ylabel("IR")

    plt.subplot(3, 1, 2)
    plt.plot(t_sec, red_list)
    plt.ylabel("RED")

    plt.subplot(3, 1, 3)
    plt.plot(t_sec, green_list)
    plt.ylabel("GREEN")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Opening serial port {PORT} at {BAUD} ...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # allow Arduino to reset

    # Clear any initial messages from Arduino (like READY)
    ser.reset_input_buffer()

    try:
        while True:
            ans = input("\nRecord a new segment? (y/n): ").strip().lower()
            if ans not in ("y", "yes"):
                break

            # Ask for label (used in filename)
            label = input("Enter a label for this segment (e.g. rest1, walk2): ").strip()
            if not label:
                label = "session"

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ppg_{label}_{timestamp}.csv"
            out_path = os.path.join(OUTPUT_DIR, filename)

            # Optionally allow per-run duration override:
            dur_in = input(f"Duration in seconds (default {DURATION_SECONDS}): ").strip()
            if dur_in:
                try:
                    duration = float(dur_in)
                except ValueError:
                    print("Invalid duration, using default.")
                    duration = DURATION_SECONDS
            else:
                duration = DURATION_SECONDS

            record_segment(ser, duration, out_path)

            # Plot right after recording
            plot_choice = input("Plot this recording now? (y/n): ").strip().lower()
            if plot_choice in ("y", "yes"):
                plot_csv(out_path)
            else:
                print(f"Skipping plot. Data saved to {out_path}")

    finally:
        ser.close()
        print("Serial closed.")


if __name__ == "__main__":
    main()
