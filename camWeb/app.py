import asyncio
import socket
import json
import functools
import threading
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import qrcode
from aiortc import RTCPeerConnection, RTCSessionDescription
from ultralytics import FastSAM

app = Flask(__name__)

# 1. Start a persistent background asyncio loop in a dedicated daemon thread
loop = asyncio.new_event_loop()

def start_background_loop(ev_loop):
    asyncio.set_event_loop(ev_loop)
    ev_loop.run_forever()

threading.Thread(target=start_background_loop, args=(loop,), daemon=True).start()

print("Loading FastSAM model...")
model = FastSAM("FastSAM-s.pt")

pcs = set()
peer_states = {}

async def consume_video(track, pc):
    """Continuously ingests video frames to keep the WebRTC pipeline alive."""
    while True:
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if pc in peer_states:
                peer_states[pc]["latest_frame"] = img
        except Exception:
            break

async def inference_loop(pc, channel):
    """Continuously runs FastSAM on the freshest frame and sends polygon coordinates."""
    while True:
        if pc.connectionState in ["closed", "failed"]:
            break
            
        state = peer_states.get(pc)
        if not state or state["latest_frame"] is None or channel.readyState != "open":
            await asyncio.sleep(0.01)
            continue
            
        # Extract frame and clear the slot to prevent processing duplicate frames
        img_to_process = state["latest_frame"].copy()
        state["latest_frame"] = None 

        h, w = img_to_process.shape[:2]

        # Offload FastSAM to a thread pool executor so it doesn't block the loop
        results = await loop.run_in_executor(
            None,
            functools.partial(
                model,
                img_to_process,
                device="cpu",      # Set to "cuda" if using an NVIDIA GPU
                retina_masks=False,
                imgsz=320,         # Keeps CPU inference responsive
                conf=0.4,
                iou=0.7,
                verbose=False
            )
        )

        polygons = []
        if results and len(results) > 0 and results[0].masks is not None:
            border_margin = 12                     # Reject objects touching within 12px of edges
            min_area = (w * h) * 0.02              # Reject objects smaller than 2% of screen
            max_area = (w * h) * 0.65              # Reject background planes > 65% of screen
            occupied_mask = np.zeros((h, w), dtype=np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Retrieve masks and bounding boxes
            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

            for idx, mask_raw in enumerate(masks_data):
                # Resize mask to frame dimensions
                mask_full = cv2.resize(mask_raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                # A. Clean rough edges and pinholes
                mask_clean = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)

                # B. Find outer contours
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # C. Filter tiny noise and large background planes
                if area < min_area or area > max_area:
                    continue

                # D. Filter out cut-off shapes touching the screen edges
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                if (x <= border_margin or y <= border_margin or 
                    (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin)):
                    continue

                # E. Prevent overlapping: reject if shape collides with an already-accepted object
                current_binary = (mask_clean > 0).astype(np.uint8)
                intersection = np.bitwise_and(occupied_mask, current_binary)
                overlap_ratio = np.sum(intersection) / float(area)

                if overlap_ratio > 0.15:  # Overlaps more than 15% with an existing shape
                    continue

                # Register this shape's area so future shapes won't overlap it
                occupied_mask = np.bitwise_or(occupied_mask, current_binary)

                # F. Simplify geometry for fast network transfer
                epsilon = 0.008 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                poly = [[round(float(pt[0][0] / w), 4), round(float(pt[0][1] / h), 4)] for pt in approx]
                if len(poly) >= 3:
                    polygons.append({
                        "points": poly,
                        "color_id": len(polygons)  # Unique ID to select distinct colors on frontend
                    })

        # Dispatch coordinates over the WebRTC DataChannel
        if channel.readyState == "open":
            try:
                channel.send(json.dumps({"polygons": polygons}))
            except Exception:
                pass
                
        await asyncio.sleep(0.01)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/offer", methods=["POST"])
def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    peer_states[pc] = {"latest_frame": None}

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            peer_states.pop(pc, None)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            loop.create_task(consume_video(track, pc))

    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "segmentation":
            loop.create_task(inference_loop(pc, channel))

    async def negotiate():
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    # Safely dispatch the WebRTC negotiation into the running background loop
    future = asyncio.run_coroutine_threadsafe(negotiate(), loop)
    try:
        response_data = future.result(timeout=15)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    url = f"http://{local_ip}:{port}"

    print("==================================================")
    print(f" DataChannel WebRTC Server Active")
    print(f" Access URL: {url}")
    print("==================================================")

    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.print_ascii(invert=True)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)