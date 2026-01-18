import os
import cv2
import mss
import csv
import sys
import time
import queue
import threading
import numpy as np
import tensorflow as tf
import pyautogui
import pyttsx3
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from pynput import keyboard

# =========================================================
# ===================== CONFIG ============================
# =========================================================
last_key_time = 0
KEY_DEBOUNCE = 0.25

MODEL_PATH = "speed_master_evade_model.keras"

IMG_SIZE = (84, 84)
SEQ_LEN = 5
FPS = 15

INITIAL_THRESHOLD = 0.45
THRESHOLD_STEP = 0.02

MONITOR = {"top": 263, "left": 337, "width": 734, "height": 869}

LOG_DIR = "logs"
ERROR_DIR = "errors"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{ERROR_DIR}/crashes", exist_ok=True)
os.makedirs(f"{ERROR_DIR}/interventions", exist_ok=True)

# =========================================================
# ===================== SAFE TTS ==========================
# =========================================================

speech_queue = queue.Queue()
last_spoken = 0

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

import subprocess
import time

_last_speech_time = 0
SPEECH_DEBOUNCE = 0.3

def speak(text):
    global _last_speech_time
    now = time.time()

    if now - _last_speech_time < SPEECH_DEBOUNCE:
        return

    _last_speech_time = now

    try:
        subprocess.Popen(
            [
                "powershell",
                "-Command",
                f'Add-Type -AssemblyName System.Speech; '
                f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except:
        pass


def speak_now(text):
    """High-priority speech (no cooldown)"""
    speech_queue.put(text)


# =========================================================
# ===================== STATE =============================
# =========================================================

EVADE_THRESHOLD = INITIAL_THRESHOLD
adaptive_enabled = True

current_action = "NO_ACTION"
last_frame = None

recent_p = deque(maxlen=30)

# =========================================================
# ===================== LOGGING ===========================
# =========================================================

def log_threshold():
    with open(f"{LOG_DIR}/thresholds.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), EVADE_THRESHOLD])

def log_p_evade(p):
    with open(f"{LOG_DIR}/p_evade.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), p, EVADE_THRESHOLD])

def log_crash(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{ERROR_DIR}/crashes/crash_{ts}.png", frame)
    with open(f"{LOG_DIR}/crashes.csv", "a", newline="") as f:
        csv.writer(f).writerow([ts, EVADE_THRESHOLD])
    speak("Crash logged")

def log_intervention(frame, key):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{ERROR_DIR}/interventions/{key}_{ts}.png", frame)
    with open(f"{LOG_DIR}/interventions.csv", "a", newline="") as f:
        csv.writer(f).writerow([ts, key, EVADE_THRESHOLD])
    speak("Manual intervention")

# =========================================================
# ===================== CONTROL ===========================
# =========================================================

def adaptive_threshold():
    global EVADE_THRESHOLD
    if adaptive_enabled and len(recent_p) == recent_p.maxlen:
        mean_p = np.mean(recent_p)
        EVADE_THRESHOLD = np.clip(mean_p + 0.08, 0.35, 0.6)

def release_all():
    pyautogui.keyUp("left")
    pyautogui.keyUp("right")

def press_action(action):
    release_all()
    if action == "LEFT":
        pyautogui.keyDown("left")
    elif action == "RIGHT":
        pyautogui.keyDown("right")

def decide_direction(frame):
    h, w = frame.shape
    return "LEFT" if frame[:, :w//2].mean() < frame[:, w//2:].mean() else "RIGHT"

def preprocess(sct):
    img = np.array(sct.grab(MONITOR))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = gray.astype(np.float32) / 255.0
    return gray[..., None], bgr

# =========================================================
# ===================== KEYBOARD ==========================
# =========================================================

def on_key_press(key):
    global EVADE_THRESHOLD, adaptive_enabled, last_key_time

    now = time.time()
    if now - last_key_time < KEY_DEBOUNCE:
        return
    last_key_time = now

    try:
        if key == keyboard.Key.up:
            adaptive_enabled = False
            EVADE_THRESHOLD = min(0.9, EVADE_THRESHOLD + THRESHOLD_STEP)
            speak(f"Threshold increased to {EVADE_THRESHOLD:.2f}")

        elif key == keyboard.Key.down:
            adaptive_enabled = False
            EVADE_THRESHOLD = max(0.1, EVADE_THRESHOLD - THRESHOLD_STEP)
            speak(f"Threshold decreased to {EVADE_THRESHOLD:.2f}")

        elif key == keyboard.Key.f12:
            if last_frame is not None:
                log_crash(last_frame)

        elif key == keyboard.Key.left or key == keyboard.Key.right:
            if last_frame is not None:
                log_intervention(last_frame, str(key))

    except Exception as e:
        print(e)
        pass


keyboard.Listener(on_press=on_key_press).start()

# =========================================================
# ===================== ANALYSIS ==========================
# =========================================================

def analyze_logs():
    print("\n📊 Running post-run analysis...")

    # ---- p_evade log must exist ----
    p_evade_path = f"{LOG_DIR}/p_evade.csv"
    if not os.path.exists(p_evade_path):
        print("⚠️ No p_evade log found — skipping analysis")
        return

    p_evade_log = np.atleast_2d(
        np.genfromtxt(p_evade_path, delimiter=",", dtype=str)
    )

    p_vals = p_evade_log[:, 1].astype(float)
    thr_vals = p_evade_log[:, 2].astype(float)

    # ---- interventions are OPTIONAL ----
    interventions_path = f"{LOG_DIR}/interventions.csv"

    if os.path.exists(interventions_path):
        interventions = np.atleast_2d(
            np.genfromtxt(interventions_path, delimiter=",", dtype=str)
        )
        human_evade = np.zeros(len(p_vals), dtype=bool)
        human_evade[:len(interventions)] = True
        print(f"✔ Interventions found: {len(interventions)}")
    else:
        human_evade = np.zeros(len(p_vals), dtype=bool)
        print("✔ No human interventions logged")

    # ---- model decisions ----
    model_evade = p_vals > thr_vals

    # ---- disagreement metrics ----
    TP = np.sum(model_evade & human_evade)
    FP = np.sum(model_evade & ~human_evade)
    FN = np.sum(~model_evade & human_evade)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    print(f"Disagreement Precision: {precision:.3f}")
    print(f"Disagreement Recall:    {recall:.3f}")

    # ---- plot p_evade vs human actions ----
    plt.figure(figsize=(10, 4))
    plt.plot(p_vals, label="p_evade")
    plt.axhline(np.mean(thr_vals), color="r", linestyle="--", label="threshold")

    if human_evade.any():
        plt.scatter(
            np.where(human_evade)[0],
            p_vals[human_evade],
            color="orange",
            label="human intervene"
        )

    plt.legend()
    plt.title("p_evade vs Human Interventions")
    plt.xlabel("Time")
    plt.ylabel("p_evade")
    plt.show()


# =========================================================
# ===================== MAIN ==============================
# =========================================================

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

sct = mss.mss()
frames = deque(maxlen=SEQ_LEN)

speak("Bot starting")
time.sleep(1)
speak("five")
time.sleep(1)
speak("four")
time.sleep(1)
speak("3")
time.sleep(1)
speak("2")
time.sleep(1)
speak("1")
time.sleep(1)

import win32gui
import win32con

WINDOW_NAME = "BOT VIEW"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 500, 300)

hwnd = win32gui.FindWindow(None, WINDOW_NAME)
win32gui.SetWindowPos(
    hwnd,
    win32con.HWND_TOPMOST,
    0, 0, 0, 0,
    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
)
speak("Game Started")
try:
    while True:
        start = time.time()

        f_gray, f_bgr = preprocess(sct)
        last_frame = f_bgr.copy()
        frames.append(f_gray)

        if len(frames) == SEQ_LEN:
            inp = np.expand_dims(np.array(frames), axis=0)
            p = model.predict(inp, verbose=0)[0][0]

            recent_p.append(p)
            adaptive_threshold()
            log_p_evade(p)

            if p > EVADE_THRESHOLD:
                d = decide_direction(f_gray.squeeze())
                if d != current_action:
                    press_action(d)
                    current_action = d
            else:
                if current_action != "NO_ACTION":
                    release_all()
                    current_action = "NO_ACTION"

            cv2.putText(
                f_bgr,
                f"p={p:.2f} thr={EVADE_THRESHOLD:.2f}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        cv2.imshow("BOT VIEW", f_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #time.sleep(max(0, (1/FPS)-(time.time()-start)))

except KeyboardInterrupt:
    pass

finally:
    try:
        keyboard.Listener.stop()
    except:
        pass

    print("\n🧠 Saving logs and analyzing...")
    log_threshold()
    release_all()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    speech_thread.join(timeout=1)

    analyze_logs()
    print("🛑 Bot stopped & analysis complete")
