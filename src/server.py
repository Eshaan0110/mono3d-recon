"""Flask web server for the 3D reconstruction viewer."""

import os
import uuid
import json
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from .config import PipelineConfig
from .pipeline import ReconstructionPipeline

app = Flask(__name__, static_folder="../web", static_url_path="")
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_BASE = Path("outputs")

# In-memory job tracking
jobs = {}


@app.route("/")
def index():
    """Serve the main web viewer."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from the web directory."""
    return send_from_directory(app.static_folder, filename)


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload a video and start reconstruction.

    Returns:
        JSON with job_id for tracking progress.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    UPLOAD_DIR.mkdir(exist_ok=True)
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    file.save(str(video_path))

    # Get quality from form data
    quality = request.form.get("quality", "medium")

    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "stage": "",
        "progress": 0.0,
        "message": "Queued for processing",
        "video_path": str(video_path),
        "output_dir": str(job_dir),
        "stats": None,
        "error": None,
    }

    # Run reconstruction in background thread
    thread = threading.Thread(
        target=_run_reconstruction,
        args=(job_id, str(video_path), str(job_dir), quality),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


def _run_reconstruction(job_id: str, video_path: str, output_dir: str,
                         quality: str):
    """Background reconstruction task."""
    def progress_cb(stage, progress, message):
        jobs[job_id].update({
            "status": "running",
            "stage": stage,
            "progress": progress,
            "message": message,
        })

    try:
        config = PipelineConfig()
        config.quality = quality

        pipeline = ReconstructionPipeline(
            config=config, progress_callback=progress_cb
        )
        stats = pipeline.run(video_path, output_dir)

        jobs[job_id].update({
            "status": "complete",
            "stage": "done",
            "progress": 1.0,
            "message": "Reconstruction complete",
            "stats": stats,
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "message": str(e),
            "error": str(e),
        })


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Get reconstruction job status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/api/model/<job_id>/<filename>")
def serve_model(job_id, filename):
    """Serve reconstructed model files (GLB, PLY)."""
    job_dir = OUTPUT_BASE / job_id
    file_path = job_dir / filename

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(str(file_path))


@app.route("/api/jobs")
def list_jobs():
    """List all reconstruction jobs."""
    return jsonify({
        jid: {"status": j["status"], "stage": j["stage"]}
        for jid, j in jobs.items()
    })


def run_server(host="0.0.0.0", port=5000, debug=False):
    """Start the Flask development server."""
    print(f"Starting server at http://{host}:{port}")
    print(f"Open http://localhost:{port} in your browser")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
