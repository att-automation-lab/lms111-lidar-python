import socket
import serial
import time
import numpy as np
from collections import deque

# -----------------------------
# LMS111 settings
# -----------------------------
LIDAR_HOST = "192.168.0.111"
LIDAR_PORT = 2111
LIDAR_CMD = b"\x02sRN LMDscandata\x03"

# -----------------------------
# Arduino / Servo settings
# -----------------------------
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

# -----------------------------
# Tracking zone for your test board
# -----------------------------
TRACK_MIN_ANGLE = -60.0
TRACK_MAX_ANGLE = 60.0
TRACK_MAX_DISTANCE = 0.40   # 30 cm
MIN_VALID_DISTANCE = 0.05   # 5 cm

# -----------------------------
# Servo settings
# -----------------------------
SERVO_CENTER = 90
SERVO_MIN = 20
SERVO_MAX = 160

SERVO_DEADBAND = 3          # เปลี่ยนน้อยกว่า 3° ไม่ส่ง
LOOP_DELAY = 0.15           # 150 ms
SMOOTH_WINDOW = 5           # ค่าเฉลี่ย 5 เฟรม

angle_history = deque(maxlen=SMOOTH_WINDOW)


def get_scan():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect((LIDAR_HOST, LIDAR_PORT))
    s.sendall(LIDAR_CMD)

    chunks = []
    while True:
        part = s.recv(8192)
        if not part:
            break
        chunks.append(part)
        if b"\x03" in part:
            break

    s.close()

    data = b"".join(chunks)
    text = data.decode("ascii", errors="ignore").strip("\x02").strip("\x03")
    tokens = text.split()

    if "DIST1" not in tokens:
        raise ValueError("DIST1 not found")

    idx = tokens.index("DIST1")
    count = int(tokens[idx + 5], 16)

    dist_hex = tokens[idx + 6: idx + 6 + count]
    dist = np.array([int(x, 16) for x in dist_hex], dtype=float) / 1000.0

    angles_deg = np.linspace(-135.0, 135.0, len(dist))
    return angles_deg, dist


def lidar_angle_to_servo(target_angle_deg):
    # กลับทิศตามที่คุณทดลองแล้วว่าถูก
    servo_angle = SERVO_CENTER - target_angle_deg

    if servo_angle < SERVO_MIN:
        servo_angle = SERVO_MIN
    if servo_angle > SERVO_MAX:
        servo_angle = SERVO_MAX

    return int(round(servo_angle))


def find_target(angles_deg, dist):
    mask = (
        (angles_deg >= TRACK_MIN_ANGLE) &
        (angles_deg <= TRACK_MAX_ANGLE) &
        (dist > MIN_VALID_DISTANCE) &
        (dist < TRACK_MAX_DISTANCE)
    )

    if not np.any(mask):
        return None, None

    candidate_angles = angles_deg[mask]
    candidate_dist = dist[mask]

    # เลือกจุดที่ใกล้สุดในโซนทดลอง
    i = np.argmin(candidate_dist)
    return float(candidate_angles[i]), float(candidate_dist[i])


def main():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")

    last_sent = None

    try:
        while True:
            angles_deg, dist = get_scan()

            target_angle, target_dist = find_target(angles_deg, dist)

            if target_angle is None:
                angle_history.clear()
                print("No target in test zone")
                time.sleep(LOOP_DELAY)
                continue

            # smoothing
            angle_history.append(target_angle)
            smooth_angle = sum(angle_history) / len(angle_history)

            servo_angle = lidar_angle_to_servo(smooth_angle)

            # deadband
            if (last_sent is None) or (abs(servo_angle - last_sent) >= SERVO_DEADBAND):
                ser.write(f"{servo_angle}\n".encode("utf-8"))
                print(
                    f"Target raw={target_angle:+.1f}° | "
                    f"smooth={smooth_angle:+.1f}° | "
                    f"dist={target_dist*100:.1f} cm | "
                    f"servo={servo_angle}"
                )
                last_sent = servo_angle

            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()