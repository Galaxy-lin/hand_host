"""Serve a Windows camera as an MJPEG stream for WSL clients.

Run this script in Windows Python, then put the shown /video URL into the
application's camera source field.
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class CameraState:
    def __init__(self, camera: int, width: int, height: int, quality: int):
        self.camera = camera
        self.width = width
        self.height = height
        self.quality = quality
        self.lock = threading.Lock()
        self.latest: bytes | None = None
        self.running = True

    def capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera}")

        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)]
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame, params)
            if ok:
                with self.lock:
                    self.latest = encoded.tobytes()
        cap.release()


class Handler(BaseHTTPRequestHandler):
    state: CameraState

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Use /video as the MJPEG camera URL.\n")
            return
        if self.path != "/video":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        while self.state.running:
            with self.state.lock:
                frame = self.state.latest
            if frame is None:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()

    state = CameraState(args.camera, args.width, args.height, args.quality)
    Handler.state = state
    thread = threading.Thread(target=state.capture_loop, daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MJPEG stream: http://{args.host}:{args.port}/video")
    try:
        server.serve_forever()
    finally:
        state.running = False
        server.server_close()


if __name__ == "__main__":
    main()
