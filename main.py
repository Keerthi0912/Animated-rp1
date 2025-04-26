#!/usr/bin/env python3
import os
import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from gpiozero import Button, LED
import pickle
import firebase_admin
from firebase_admin import credentials, firestore, storage
from signal import pause
from datetime import datetime

# Configuration
INPUT_IMAGE    = "test.jpg"
OUTPUT_IMAGE   = "ghibli_output.png"
GRID_SIZE      = 4
MODEL_FILE     = "saved_model.pb"
STORAGE_FOLDER = "rpi_captures"
BUCKET_NAME    = "kvanaparthi-sgopaluni1cmsc5313.firebasestorage.app"
RED_LED_PIN    = 22
BLUE_LED_PIN   = 16
GREEN_LED_PIN  = 27
BUTTON_PIN     = 19

# Initialize Firebase
cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {"storageBucket": BUCKET_NAME})
db     = firestore.client()
bucket = storage.bucket()
doc_project_ref = db.collection("RPI_DATA").document("DOC_PROJECT")

# Initialize hardware
red_led   = LED(RED_LED_PIN)
blue_led  = LED(BLUE_LED_PIN)
green_led = LED(GREEN_LED_PIN)
button    = Button(BUTTON_PIN)
picam2    = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (800, 600)})
picam2.configure(camera_config)

def apply_edge_enhancement(arr):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    enhanced = np.zeros_like(arr)
    for i in range(1, arr.shape[0]-1):
        for j in range(1, arr.shape[1]-1):
            for ch in range(3):
                enhanced[i,j,ch] = np.sum(arr[i-1:i+2, j-1:j+2, ch] * kernel)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def detect_missing_colors(image: Image.Image):
    arr = np.array(image)
    means = arr.reshape(-1,3).mean(axis=0)
    missing = []
    if means[0] < 50: missing.append("red")
    if means[1] < 50: missing.append("green")
    if means[2] < 50: missing.append("blue")
    return missing

def upload_to_storage(local_path: str, target_folder: str, remote_filename: str = None) -> str:
    """Upload a local file and return its public URL."""
    fname = remote_filename or os.path.basename(local_path)
    blob = bucket.blob(f"{target_folder}/{fname}")
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url

def run_style_transfer(input_img_path: str):
    img = Image.open(input_img_path).convert("RGB")
    w, h = img.size
    cw, ch = w // GRID_SIZE, h // GRID_SIZE
    canvas = Image.new("RGB", img.size)

    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x, y = c * cw, r * ch
                patch = img.crop((x, y, x+cw, y+ch)).resize((256,256))
                patch_array = np.array(patch, np.float32) / 255.0
                styled_array = model.predict(patch_array[None, ...])[0]
                styled = Image.fromarray((np.clip(styled_array,0,1)*255).astype(np.uint8))
                styled = styled.resize((cw, ch))
                arr = np.array(styled, float)
                enhanced = apply_edge_enhancement(arr)
                canvas.paste(Image.fromarray(enhanced), (x, y))

        canvas.save(OUTPUT_IMAGE)
        return True, canvas

    except Exception as e:
        print(f"Model loading failed: {e}")
        return False, None

def process_fallback(input_img_path: str):
    img = Image.open(input_img_path).convert("RGB")
    w, h = img.size
    cw, ch = w // GRID_SIZE, h // GRID_SIZE
    canvas = Image.new("RGB", img.size)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * cw, r * ch
            patch = img.crop((x, y, x+cw, y+ch))
            arr = np.array(patch, np.float32)
            arr[:,:,0] = np.clip(arr[:,:,0] * 1.2, 0, 255)  # boost red
            arr[:,:,2] = np.clip(arr[:,:,2] * 0.8, 0, 255)  # reduce blue
            arr = (arr - 128) * 1.2 + 128
            arr = np.clip(arr, 0, 255)
            enhanced = apply_edge_enhancement(arr)
            canvas.paste(Image.fromarray(enhanced.astype(np.uint8)), (x, y))

    canvas.save(OUTPUT_IMAGE)
    return canvas

def on_button_press():
    # create a human-readable timestamp for filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Capturing image at {ts}…")

    picam2.start()
    time.sleep(0.2)
    picam2.capture_file(INPUT_IMAGE)
    picam2.stop()
    print(f"Saved {INPUT_IMAGE}")

    # Stylize (or fallback)
    success, output_img = run_style_transfer(INPUT_IMAGE)
    if not success:
        print("Stylization failed, using fallback.")
        output_img = process_fallback(INPUT_IMAGE)

    # Detect missing colors & drive LEDs
    missing = detect_missing_colors(output_img)
    print("Missing colors:", missing)
    red_led.on()   if "red"   in missing else red_led.off()
    green_led.on() if "green" in missing else green_led.off()
    blue_led.on()  if "blue"  in missing else blue_led.off()

    # Upload both images with unique filenames
    orig_url = upload_to_storage(INPUT_IMAGE,  STORAGE_FOLDER, f"input_{ts}.jpg")
    anim_url = upload_to_storage(OUTPUT_IMAGE, STORAGE_FOLDER, f"output_{ts}.png")
    print("Uploaded to storage:", orig_url, anim_url)

    # get a timezone-aware datetime (local zone)
    now = datetime.now().astimezone()

    # Overwrite Firestore fields with the new URLs & metadata
    doc_project_ref.update({
        "originalPicUrl":  orig_url,
        "animationPicUrl": anim_url,
        "color":           missing,
        "timestamp":       now,
    })
    print("Firestore document updated. ✅")

# Hook up the button & start listening (nothing runs until you press)
button.when_pressed = on_button_press
print("Ready — press the button to capture & stylize.")
pause()
