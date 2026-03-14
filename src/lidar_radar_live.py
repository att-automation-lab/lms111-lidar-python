import socket
import numpy as np
import matplotlib.pyplot as plt

HOST = "192.168.0.102"
PORT = 2111
CMD = b"\x02sRN LMDscandata\x03"


def get_scan():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect((HOST, PORT))
    s.sendall(CMD)

    chunks = []
    while True:
        part = s.recv(8192)
        if not part:
            break
        chunks.append(part)
        if b"\x03" in part:   # เจอ ETX
            break

    s.close()

    data = b"".join(chunks)
    text = data.decode("ascii", errors="ignore").strip("\x02").strip("\x03")
    tokens = text.split()

    if "DIST1" not in tokens:
        raise ValueError("DIST1 not found in telegram")

    idx = tokens.index("DIST1")
    data_count = int(tokens[idx + 5], 16)

    distances_hex = tokens[idx + 6: idx + 6 + data_count]
    distances_m = np.array([int(x, 16) for x in distances_hex], dtype=float) / 1000.0

    # สร้างมุมตามจำนวนจุดจริงของรอบนั้น
    angles_deg = np.linspace(-135.0, 135.0, len(distances_m))
    angles_rad = np.deg2rad(angles_deg)

    return angles_rad, distances_m


plt.ion()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="polar")
ax.set_title("LMS111 Real-Time Radar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(-135)
ax.set_thetamax(135)
ax.set_rmax(5)
ax.grid(True)

# เริ่มด้วย array ว่าง
line, = ax.plot([], [], marker=".", linestyle="None")

try:
    while True:
        angles, distances = get_scan()

        # กันค่าผิดปกติ
        n = min(len(angles), len(distances))
        angles = angles[:n]
        distances = distances[:n]

        line.set_data(angles, distances)
        fig.canvas.draw_idle()
        plt.pause(0.05)

except KeyboardInterrupt:
    print("Stopped by user.")
except Exception as e:
    print("ERROR:", e)