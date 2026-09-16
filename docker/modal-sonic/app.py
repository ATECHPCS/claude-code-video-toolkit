"""
Modal deployment for Sonic talking head generation.
(Tencent/Zhejiang "Sonic: Shifting Focus to Global Audio Perception in
Portrait Animation" — the 2025-generation upgrade over SadTalker.)

Deploy:
    modal deploy docker/modal-sonic/app.py

Weights (~25GB: Sonic + SVD-xt + whisper-tiny, all ungated) are pulled into a
Modal Volume on first container start and cached across runs. First cold start
therefore takes several minutes; subsequent ones just load to GPU.

Generation speed: roughly 20-40s of GPU time per second of audio on L40S —
budget accordingly for long narrations (a 76s track is ~$1-2, not pennies).
"""

import modal

app = modal.App("video-toolkit-sonic")

WEIGHTS_DIR = "/weights"
SONIC_DIR = "/app/Sonic"

volume = modal.Volume.from_name("sonic-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    # Sonic's pinned requirements (gradio omitted — API only)
    .pip_install(
        "torch==2.2.1", "torchvision==0.17.1", "torchaudio==2.2.1",
        "diffusers==0.29.0", "transformers==4.43.2",
        "imageio==2.31.1", "imageio-ffmpeg==0.5.1",
        "omegaconf==2.3.0", "tqdm==4.65.2",
        "librosa==0.10.2.post1", "einops==0.7.0",
        "opencv-python-headless", "scipy", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "fastapi[standard]",
        "requests", "boto3",
    )
    # torch 2.2.1 is compiled against numpy 1.x — a transitive dep pulls in
    # numpy 2.x, which torch.from_numpy rejects ("Numpy is not available").
    # Final pin wins the resolution.
    .pip_install("numpy<2")
    .run_commands(f"git clone --depth 1 https://github.com/jixiaozhong/Sonic.git {SONIC_DIR}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": SONIC_DIR})
)


def _ensure_weights():
    """Download all checkpoints into the volume (idempotent), then symlink
    them into the layout Sonic's config expects (checkpoints/ under repo)."""
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    marker = Path(WEIGHTS_DIR) / ".complete"
    if not marker.exists():
        snapshot_download(
            "LeonJoe13/Sonic",
            local_dir=f"{WEIGHTS_DIR}/sonic",
            ignore_patterns=["*.mp4", "*.gif", "*.png"],
        )
        snapshot_download(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            local_dir=f"{WEIGHTS_DIR}/svd-xt",
            ignore_patterns=["*.mp4", "*.gif", "*.bin", "*onnx*"],
        )
        snapshot_download("openai/whisper-tiny", local_dir=f"{WEIGHTS_DIR}/whisper-tiny")
        marker.touch()
        volume.commit()

    ckpt = Path(SONIC_DIR) / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    # LeonJoe13/Sonic mirrors the checkpoints/ root: Sonic/, RIFE/, yoloface_v5m.pt
    for item in Path(f"{WEIGHTS_DIR}/sonic").iterdir():
        if item.name.startswith("."):
            continue
        dst = ckpt / item.name
        if not dst.exists():
            os.symlink(item, dst)
    for src, name in [
        (f"{WEIGHTS_DIR}/svd-xt", "stable-video-diffusion-img2vid-xt"),
        (f"{WEIGHTS_DIR}/whisper-tiny", "whisper-tiny"),
    ]:
        dst = ckpt / name
        if not dst.exists():
            os.symlink(src, dst)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={WEIGHTS_DIR: volume},
    timeout=7200,
    scaledown_window=120,
    memory=32768,
    cpu=8,
)
@modal.concurrent(max_inputs=1)
class SonicGen:
    @modal.enter()
    def load(self):
        import os

        import torch

        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        _ensure_weights()
        os.chdir(SONIC_DIR)  # config paths are relative to the repo
        from sonic import Sonic

        self.pipe = Sonic(0)
        print("Sonic pipeline loaded")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        import base64
        import shutil
        import subprocess
        import tempfile
        import time
        import uuid
        from pathlib import Path

        import requests as req

        start_time = time.time()

        image_url = request.get("image_url")
        image_base64 = request.get("image_base64")
        audio_url = request.get("audio_url")
        audio_base64 = request.get("audio_base64")
        if not image_url and not image_base64:
            return {"error": "Missing image_url or image_base64"}
        if not audio_url and not audio_base64:
            return {"error": "Missing audio_url or audio_base64"}

        dynamic_scale = float(request.get("dynamic_scale", 1.0))
        # lip articulation strength (config default 7.5). Lower = subtler
        # mouth, less teeth; higher = harder enunciation.
        audio_guidance = request.get("audio_guidance_scale")
        crop = bool(request.get("crop", False))
        min_resolution = int(request.get("min_resolution", 512))
        inference_steps = int(request.get("inference_steps", 25))
        r2_config = request.get("r2")

        work = Path(tempfile.mkdtemp(prefix="modal_sonic_"))
        try:
            def fetch(url, b64, path):
                if url:
                    r = req.get(url, stream=True, timeout=300)
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                else:
                    data = b64.split(",", 1)[1] if "," in b64 else b64
                    path.write_bytes(base64.b64decode(data))

            image_path = work / "input_image.png"
            audio_path = work / "input_audio.wav"
            fetch(image_url, image_base64, image_path)
            fetch(audio_url, audio_base64, audio_path)

            duration = 0.0
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                    capture_output=True, text=True, timeout=30,
                )
                duration = float(out.stdout.strip())
            except Exception:
                pass
            print(f"Audio: {duration:.1f}s, dynamic_scale={dynamic_scale}, crop={crop}")

            face_info = self.pipe.preprocess(str(image_path), expand_ratio=0.5)
            print(f"Face info: {face_info}")
            if face_info.get("face_num", 0) <= 0:
                return {"error": "No face detected in image"}
            src = str(image_path)
            if crop:
                cropped = str(image_path) + ".crop.png"
                self.pipe.crop_image(str(image_path), cropped, face_info["crop_bbox"])
                src = cropped

            # pipe.process reads config values at call time — safe to mutate,
            # but always reset so warm containers don't inherit prior requests
            self.pipe.config.audio_guidance_scale = (
                float(audio_guidance) if audio_guidance is not None else 7.5
            )
            # expression-injection strength: <1.0 damps audio-driven smiles /
            # emotive energy without touching lip-sync timing
            self.pipe.config.ip_audio_scale = float(request.get("ip_audio_scale", 1.0))
            print(f"audio_guidance_scale={self.pipe.config.audio_guidance_scale} "
                  f"ip_audio_scale={self.pipe.config.ip_audio_scale}")

            out_path = work / "output" / "result.mp4"
            out_path.parent.mkdir()
            self.pipe.process(
                src, str(audio_path), str(out_path),
                min_resolution=min_resolution,
                inference_steps=inference_steps,
                dynamic_scale=dynamic_scale,
            )
            if not out_path.exists():
                # some versions write <output>/<name>.mp4 variants
                found = list(out_path.parent.glob("*.mp4"))
                if not found:
                    return {"error": "Sonic produced no output video"}
                out_path = found[0]

            elapsed = time.time() - start_time
            print(f"Done: {elapsed:.1f}s for {duration:.1f}s audio")
            result = {
                "success": True,
                "duration_seconds": round(duration, 2),
                "processing_time_seconds": round(elapsed, 2),
            }

            if r2_config:
                import boto3
                from botocore.config import Config

                client = boto3.client(
                    "s3",
                    endpoint_url=r2_config["endpoint_url"],
                    aws_access_key_id=r2_config["access_key_id"],
                    aws_secret_access_key=r2_config["secret_access_key"],
                    config=Config(signature_version="s3v4"),
                )
                key = f"sonic/results/{uuid.uuid4().hex[:12]}.mp4"
                client.upload_file(
                    str(out_path), r2_config["bucket_name"], key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                result["video_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": r2_config["bucket_name"], "Key": key},
                    ExpiresIn=7200,
                )
                result["r2_key"] = key
            else:
                result["video_base64"] = base64.b64encode(out_path.read_bytes()).decode()

            return result
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return {"error": f"Internal error: {e}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
