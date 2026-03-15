import socket
import serial
import time
import threading
from collections import deque

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

# =============================
# SETTINGS
# =============================
LIDAR_HOST = "192.168.0.111"
LIDAR_PORT = 2111
LIDAR_CMD = b"\x02sRN LMDscandata\x03"

SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

DEFAULT_TRACK_MIN_ANGLE = -60.0
DEFAULT_TRACK_MAX_ANGLE = 60.0
DEFAULT_TRACK_MAX_DISTANCE = 0.30   # 30 cm
DEFAULT_MIN_VALID_DISTANCE = 0.05   # 5 cm

SERVO_CENTER = 90
SERVO_MIN = 20
SERVO_MAX = 160

SERVO_DEADBAND = 3
LOOP_DELAY = 0.15
SMOOTH_WINDOW = 5

X_LIM = 0.40
Y_LIM = 0.40

# =============================
# GLOBALS
# =============================
running = False
ser = None
worker_thread = None
angle_history = deque(maxlen=SMOOTH_WINDOW)
last_sent = None
baseline = None

latest_scan_x = np.array([])
latest_scan_y = np.array([])
latest_hit_x = np.array([])
latest_hit_y = np.array([])
latest_target_x = np.array([])
latest_target_y = np.array([])

status_text = "IDLE"
target_angle_text = "-"
target_dist_text = "-"
servo_angle_text = "-"
mode_text = "LIVE"

data_lock = threading.Lock()

# runtime zone values
track_min_angle = DEFAULT_TRACK_MIN_ANGLE
track_max_angle = DEFAULT_TRACK_MAX_ANGLE
track_max_distance = DEFAULT_TRACK_MAX_DISTANCE
min_valid_distance = DEFAULT_MIN_VALID_DISTANCE


# =============================
# LIDAR FUNCTIONS
# =============================
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

    # LMS111 raw angle model we used before
    angles_deg = np.linspace(-135.0, 135.0, len(dist))
    return angles_deg, dist


def polar_to_cartesian_top_up(angles_deg, dist):
    """
    เปลี่ยนให้ sensor อยู่ล่าง และสแกนขึ้นบน
    เดิม 0° = แกน +X
    ใหม่ 0° = แกน +Y
    """
    rad = np.deg2rad(angles_deg)
    x = dist * np.sin(rad)
    y = dist * np.cos(rad)
    return x, y


def lidar_angle_to_servo(target_angle_deg):
    servo_angle = SERVO_CENTER - target_angle_deg
    servo_angle = max(SERVO_MIN, min(SERVO_MAX, servo_angle))
    return int(round(servo_angle))


def find_target(angles_deg, dist):
    mask = (
        (angles_deg >= track_min_angle) &
        (angles_deg <= track_max_angle) &
        (dist > min_valid_distance) &
        (dist < track_max_distance)
    )

    if not np.any(mask):
        return None, None, mask

    candidate_angles = angles_deg[mask]
    candidate_dist = dist[mask]

    i = np.argmin(candidate_dist)
    return float(candidate_angles[i]), float(candidate_dist[i]), mask


def teach_baseline(frames=8):
    global baseline, mode_text
    scans = []
    mode_text = "TEACHING BASELINE"
    for i in range(frames):
        angles_deg, dist = get_scan()
        scans.append(dist)
        time.sleep(0.05)
    baseline = np.median(np.array(scans), axis=0)
    mode_text = "BASELINE READY"


# =============================
# SERIAL FUNCTIONS
# =============================
def open_serial():
    global ser
    if ser is None or not ser.is_open:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)


def close_serial():
    global ser
    try:
        if ser is not None and ser.is_open:
            ser.close()
    except Exception:
        pass


def send_servo(angle):
    global last_sent
    if ser is not None and ser.is_open:
        ser.write(f"{angle}\n".encode("utf-8"))
        last_sent = angle


# =============================
# WORKER
# =============================
def tracking_loop(use_baseline=False):
    global running, last_sent
    global latest_scan_x, latest_scan_y
    global latest_hit_x, latest_hit_y
    global latest_target_x, latest_target_y
    global status_text, target_angle_text, target_dist_text, servo_angle_text, mode_text

    try:
        open_serial()
    except Exception as e:
        messagebox.showerror("Serial Error", f"Open {SERIAL_PORT} failed:\n{e}")
        running = False
        return

    if use_baseline:
        try:
            teach_baseline(frames=8)
        except Exception as e:
            messagebox.showerror("Baseline Error", str(e))
            running = False
            return

    while running:
        try:
            angles_deg, dist = get_scan()
            x, y = polar_to_cartesian_top_up(angles_deg, dist)

            valid_scan = dist > min_valid_distance
            scan_x = x[valid_scan]
            scan_y = y[valid_scan]

            target_angle, target_dist, zone_mask = find_target(angles_deg, dist)

            hit_x = x[zone_mask]
            hit_y = y[zone_mask]

            if baseline is not None and len(baseline) == len(dist):
                delta = baseline - dist
                new_obj_mask = zone_mask & (delta > 0.03)

                if np.any(new_obj_mask):
                    hit_x = x[new_obj_mask]
                    hit_y = y[new_obj_mask]
                    candidate_angles = angles_deg[new_obj_mask]
                    candidate_dist = dist[new_obj_mask]
                    i = np.argmin(candidate_dist)
                    target_angle = float(candidate_angles[i])
                    target_dist = float(candidate_dist[i])

            if target_angle is None:
                angle_history.clear()
                with data_lock:
                    latest_scan_x = scan_x
                    latest_scan_y = scan_y
                    latest_hit_x = np.array([])
                    latest_hit_y = np.array([])
                    latest_target_x = np.array([])
                    latest_target_y = np.array([])
                    status_text = "NO TARGET"
                    target_angle_text = "-"
                    target_dist_text = "-"
                    servo_angle_text = "-"
                time.sleep(LOOP_DELAY)
                continue

            angle_history.append(target_angle)
            smooth_angle = sum(angle_history) / len(angle_history)
            servo_angle = lidar_angle_to_servo(smooth_angle)

            if (last_sent is None) or (abs(servo_angle - last_sent) >= SERVO_DEADBAND):
                send_servo(servo_angle)

            tx, ty = polar_to_cartesian_top_up(np.array([target_angle]), np.array([target_dist]))

            with data_lock:
                latest_scan_x = scan_x
                latest_scan_y = scan_y
                latest_hit_x = hit_x
                latest_hit_y = hit_y
                latest_target_x = tx
                latest_target_y = ty
                status_text = "TRACKING"
                target_angle_text = f"{smooth_angle:+.1f} deg"
                target_dist_text = f"{target_dist*100:.1f} cm"
                servo_angle_text = f"{servo_angle} deg"

        except Exception as e:
            with data_lock:
                status_text = f"ERROR: {e}"

        time.sleep(LOOP_DELAY)

    close_serial()


# =============================
# GUI ACTIONS
# =============================
def apply_zone_from_gui():
    global track_min_angle, track_max_angle, track_max_distance, min_valid_distance

    try:
        a1 = float(entry_min_angle.get())
        a2 = float(entry_max_angle.get())
        dmax_cm = float(entry_max_dist_cm.get())
        dmin_cm = float(entry_min_dist_cm.get())

        if a1 >= a2:
            messagebox.showerror("Zone Error", "Min angle must be less than Max angle")
            return

        track_min_angle = a1
        track_max_angle = a2
        track_max_distance = dmax_cm / 100.0
        min_valid_distance = dmin_cm / 100.0

        update_zone_patch()
        messagebox.showinfo("Zone", "Zone updated")
    except Exception as e:
        messagebox.showerror("Zone Error", str(e))


def update_zone_patch():
    # sensor อยู่ล่าง ยิงขึ้นบน -> ต้องเลื่อนมุมของ wedge จาก +Y มาเป็น +X reference
    theta1 = 90 - track_max_angle
    theta2 = 90 - track_min_angle

    zone.set_center((0, 0))
    zone.r = track_max_distance
    zone.theta1 = theta1
    zone.theta2 = theta2

    line_left.set_data(
        [0, track_max_distance * np.sin(np.deg2rad(track_min_angle))],
        [0, track_max_distance * np.cos(np.deg2rad(track_min_angle))]
    )
    line_right.set_data(
        [0, track_max_distance * np.sin(np.deg2rad(track_max_angle))],
        [0, track_max_distance * np.cos(np.deg2rad(track_max_angle))]
    )


def start_tracking():
    global running, worker_thread, mode_text
    if running:
        return
    running = True
    mode_text = "LIVE"
    worker_thread = threading.Thread(target=tracking_loop, args=(False,), daemon=True)
    worker_thread.start()


def start_with_baseline():
    global running, worker_thread, mode_text
    if running:
        return
    running = True
    mode_text = "START WITH BASELINE"
    worker_thread = threading.Thread(target=tracking_loop, args=(True,), daemon=True)
    worker_thread.start()


def stop_tracking():
    global running, status_text
    running = False
    status_text = "STOPPED"


def teach_background():
    try:
        stop_tracking()
        teach_baseline(frames=8)
        messagebox.showinfo("Baseline", "Baseline stored successfully.")
    except Exception as e:
        messagebox.showerror("Baseline Error", str(e))


def reset_servo_center():
    try:
        open_serial()
        send_servo(SERVO_CENTER)
        messagebox.showinfo("Servo", f"Servo reset to center = {SERVO_CENTER}")
    except Exception as e:
        messagebox.showerror("Servo Error", str(e))


def manual_left():
    try:
        open_serial()
        send_servo(60)
    except Exception as e:
        messagebox.showerror("Servo Error", str(e))


def manual_center():
    try:
        open_serial()
        send_servo(SERVO_CENTER)
    except Exception as e:
        messagebox.showerror("Servo Error", str(e))


def manual_right():
    try:
        open_serial()
        send_servo(120)
    except Exception as e:
        messagebox.showerror("Servo Error", str(e))


def update_plot():
    with data_lock:
        sc_all.set_offsets(np.column_stack((latest_scan_x, latest_scan_y)) if len(latest_scan_x) else np.empty((0, 2)))
        sc_hit.set_offsets(np.column_stack((latest_hit_x, latest_hit_y)) if len(latest_hit_x) else np.empty((0, 2)))
        sc_target.set_offsets(np.column_stack((latest_target_x, latest_target_y)) if len(latest_target_x) else np.empty((0, 2)))

        if len(latest_target_x):
            target_line.set_data([0, latest_target_x[0]], [0, latest_target_y[0]])
        else:
            target_line.set_data([], [])

        lbl_status_val.config(text=status_text)
        lbl_angle_val.config(text=target_angle_text)
        lbl_dist_val.config(text=target_dist_text)
        lbl_servo_val.config(text=servo_angle_text)
        lbl_mode_val.config(text=mode_text)
        lbl_ip_val.config(text=LIDAR_HOST)
        lbl_com_val.config(text=SERIAL_PORT)

        if status_text == "TRACKING":
            zone.set_facecolor("red")
            zone.set_edgecolor("red")
            zone.set_alpha(0.22)
            ax.set_title("LMS111 GUI Tracking | TARGET DETECTED", color="red")
        else:
            zone.set_facecolor("green")
            zone.set_edgecolor("green")
            zone.set_alpha(0.12)
            ax.set_title("LMS111 GUI Tracking", color="green")

    canvas.draw_idle()
    root.after(100, update_plot)


def on_close():
    stop_tracking()
    close_serial()
    root.destroy()


# =============================
# GUI BUILD
# =============================
root = tk.Tk()
root.title("LMS111 LiDAR GUI Tracking")
root.geometry("1280x820")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=8, pady=8)

left_frame = ttk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = ttk.Frame(main_frame, width=320)
right_frame.pack(side="right", fill="y", padx=10)

# Figure
fig = Figure(figsize=(7.5, 7.5), dpi=100)
ax = fig.add_subplot(111)
ax.set_xlim(-X_LIM, X_LIM)
ax.set_ylim(-0.02, Y_LIM)   # sensor อยู่ล่าง
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("LMS111 GUI Tracking", color="green")

# sensor position
ax.scatter([0], [0], marker="x", s=120, label="LMS111")

# zone
zone = Wedge(
    center=(0, 0),
    r=track_max_distance,
    theta1=90 - track_max_angle,
    theta2=90 - track_min_angle,
    facecolor="green",
    alpha=0.12,
    edgecolor="green",
    linewidth=2,
)
ax.add_patch(zone)

line_left, = ax.plot([], [], "g--", linewidth=1)
line_right, = ax.plot([], [], "g--", linewidth=1)
update_zone_patch()

# scatter layers
sc_all = ax.scatter([], [], s=8, label="Scan points")
sc_hit = ax.scatter([], [], s=22, label="Zone points")
sc_target = ax.scatter([], [], s=90, marker="o", label="Target")
target_line, = ax.plot([], [], linewidth=2)

ax.legend(loc="upper right")

canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Right panel
ttk.Label(right_frame, text="Status Panel", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

def make_row(label_text):
    row = ttk.Frame(right_frame)
    row.pack(fill="x", pady=3)
    ttk.Label(row, text=label_text, width=14).pack(side="left")
    val = ttk.Label(row, text="-")
    val.pack(side="left")
    return val

lbl_status_val = make_row("Status")
lbl_angle_val = make_row("Target angle")
lbl_dist_val = make_row("Target dist")
lbl_servo_val = make_row("Servo angle")
lbl_mode_val = make_row("Mode")
lbl_ip_val = make_row("LiDAR IP")
lbl_com_val = make_row("COM Port")

ttk.Separator(right_frame).pack(fill="x", pady=10)

ttk.Label(right_frame, text="Zone Settings", font=("Segoe UI", 11, "bold")).pack(anchor="w")

zone_frame = ttk.Frame(right_frame)
zone_frame.pack(fill="x", pady=5)

ttk.Label(zone_frame, text="Min angle").grid(row=0, column=0, sticky="w")
entry_min_angle = ttk.Entry(zone_frame, width=10)
entry_min_angle.grid(row=0, column=1, padx=4, pady=2)
entry_min_angle.insert(0, str(int(DEFAULT_TRACK_MIN_ANGLE)))

ttk.Label(zone_frame, text="Max angle").grid(row=1, column=0, sticky="w")
entry_max_angle = ttk.Entry(zone_frame, width=10)
entry_max_angle.grid(row=1, column=1, padx=4, pady=2)
entry_max_angle.insert(0, str(int(DEFAULT_TRACK_MAX_ANGLE)))

ttk.Label(zone_frame, text="Min dist cm").grid(row=2, column=0, sticky="w")
entry_min_dist_cm = ttk.Entry(zone_frame, width=10)
entry_min_dist_cm.grid(row=2, column=1, padx=4, pady=2)
entry_min_dist_cm.insert(0, str(int(DEFAULT_MIN_VALID_DISTANCE * 100)))

ttk.Label(zone_frame, text="Max dist cm").grid(row=3, column=0, sticky="w")
entry_max_dist_cm = ttk.Entry(zone_frame, width=10)
entry_max_dist_cm.grid(row=3, column=1, padx=4, pady=2)
entry_max_dist_cm.insert(0, str(int(DEFAULT_TRACK_MAX_DISTANCE * 100)))

ttk.Button(right_frame, text="Apply Zone", command=apply_zone_from_gui).pack(fill="x", pady=4)

ttk.Separator(right_frame).pack(fill="x", pady=10)

ttk.Button(right_frame, text="Start", command=start_tracking).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Start with Baseline", command=start_with_baseline).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Stop", command=stop_tracking).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Teach Baseline", command=teach_background).pack(fill="x", pady=4)

ttk.Separator(right_frame).pack(fill="x", pady=10)

ttk.Label(right_frame, text="Manual Servo", font=("Segoe UI", 11, "bold")).pack(anchor="w")
ttk.Button(right_frame, text="Reset Servo Center", command=reset_servo_center).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Manual Left", command=manual_left).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Manual Center", command=manual_center).pack(fill="x", pady=4)
ttk.Button(right_frame, text="Manual Right", command=manual_right).pack(fill="x", pady=4)

ttk.Separator(right_frame).pack(fill="x", pady=10)
ttk.Button(right_frame, text="Exit", command=on_close).pack(fill="x", pady=4)

tips = (
    "Layout:\n"
    "- Sensor at bottom\n"
    "- Scan upward\n\n"
    "Current defaults:\n"
    f"Angle: {DEFAULT_TRACK_MIN_ANGLE:.0f} to {DEFAULT_TRACK_MAX_ANGLE:.0f} deg\n"
    f"Distance: {DEFAULT_MIN_VALID_DISTANCE*100:.0f} to {DEFAULT_TRACK_MAX_DISTANCE*100:.0f} cm\n\n"
    "Use Start with Baseline when board is empty."
)
ttk.Label(right_frame, text=tips, justify="left").pack(anchor="w", pady=14)

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, update_plot)
root.mainloop()