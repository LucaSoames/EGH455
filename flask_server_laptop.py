# Flask server on the laptop (choose your path)
from flask import Flask, request, Response, jsonify, render_template_string
import threading
import time

app = Flask(__name__)

last_frame = None
last_frame_lock = threading.Lock()
last_telemetry = {"timestamp": "", "aruco_ids": [], "ball_gauge": None, "gauge_pressure": None,
                  "target_pressure": 50.0, "ready_to_drill": False, "env": {}}
command = {"drill": False}
status_lock = threading.Lock()

@app.route("/")
def index():
    return render_template_string("""
    <html><head><title>Drone Dashboard</title></head>
    <body>
      <h2>Live Stream</h2>
      <img src="/stream" style="width:640px;height:480px;border:1px solid #333"/>
      <h3>Status</h3>
      <pre id="status"></pre>
      <button onclick="drill()">Trigger Drill</button>
      <script>
        async function refresh() {
          const r = await fetch('/status'); 
          document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
        }
        async function drill() {
          await fetch('/command/drill', {method:'POST'});
          await refresh();
        }
        setInterval(refresh, 1000);
        refresh();
      </script>
    </body></html>
    """)

@app.route("/frame", methods=["POST"])
def frame():
    global last_frame
    data = request.get_data()
    if not data:
      return "no data", 400
    with last_frame_lock:
      last_frame = data
    return "ok"

@app.route("/telemetry", methods=["POST"])
def telemetry():
    global last_telemetry
    data = request.get_json(silent=True) or {}
    with status_lock:
        last_telemetry.update(data)
        # Optionally set ready_to_drill based on gauge pressure
        gp = data.get("gauge_pressure")
        tp = last_telemetry.get("target_pressure", 50.0)
        last_telemetry["ready_to_drill"] = (gp is not None and gp >= tp)
    return jsonify({"status": "ok"})

@app.route("/status")
def status():
    with status_lock:
        return jsonify(last_telemetry)

@app.route("/stream")
def stream():
    def gen():
        while True:
            with last_frame_lock:
                buf = last_frame
            if buf:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/command/drill", methods=["POST"])
def cmd_drill():
    with status_lock:
        command["drill"] = True
    return jsonify({"status": "queued"})

@app.route("/control")
def control():
    with status_lock:
        d = command["drill"]
        command["drill"] = False
    return jsonify({"drill": d})

if __name__ == "__main__":
    print("Server listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)