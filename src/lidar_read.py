import socket
import numpy as np
import matplotlib.pyplot as plt

HOST = "192.168.0.102"
PORT = 2111

cmd = b"\x02sRN LMDscandata\x03"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.sendall(cmd)
data = s.recv(8192)

s.close()

text = data.decode("ascii", errors="ignore").strip("\x02").strip("\x03")
tokens = text.split()

idx = tokens.index("DIST1")

data_count = int(tokens[idx + 5], 16)

distances_hex = tokens[idx + 6 : idx + 6 + data_count]
distances_mm = np.array([int(x, 16) for x in distances_hex])

# แปลง mm → meter
distances = distances_mm / 1000.0

# สร้างมุม
angles = np.linspace(-135, 135, len(distances))
angles_rad = np.deg2rad(angles)

# แปลง polar → cartesian
x = distances * np.cos(angles_rad)
y = distances * np.sin(angles_rad)

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=5)

plt.title("LMS111 LiDAR Scan")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid()

plt.show()