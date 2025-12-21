import os
import uuid
import logging
import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client

# Try to import your wrapper. If it fails, app will still start but endpoint will raise clear error.
try:
    from retinova_cli import RetiNovaCLI as WrapperClass
except Exception as e:
    WrapperClass = None    
    wrapper_import_error = e

# ----- logging -----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retina_app")

# ----- load env -----
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_USER_IMAGES = os.getenv("BUCKET_USER_IMAGES", "user-images")
BUCKET_OVERLAY = os.getenv("BUCKET_OVERLAY", "overlay-images")
BUCKET_HEATMAP = os.getenv("BUCKET_HEATMAP", "heatmaps")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "model")
MODEL_OBJECT_PATH = os.getenv("MODEL_OBJECT_PATH", "retinova_model.h5")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY missing in .env")

# Create supabase client (server side)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- FastAPI app -----
app = FastAPI(title="Retina (7-disease) API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- local folders -----
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ----- helper wrappers for supabase responses -----
def _raise_if_response_error(resp, context=""):
    if getattr(resp, "error", None):
        raise RuntimeError(f"{context} error: {resp.error}")

def _signed_url_from_bucket(bucket: str, path: str, expires_sec: int = 3600):
    resp = supabase.storage.from_(bucket).create_signed_url(path, expires_sec)
    _raise_if_response_error(resp, f"create_signed_url({bucket}/{path})")
    if getattr(resp, "data", None):
        return resp.data.get("signedURL")
    return None

def _upload_bytes_to_bucket(bucket: str, path: str, b: bytes, content_type: str = "image/png"):
    resp = supabase.storage.from_(bucket).upload(path, b, {"content-type": content_type})
    _raise_if_response_error(resp, f"upload({bucket}/{path})")
    return True

def _insert_table(table_name: str, payload: dict):
    resp = supabase.table(table_name).insert(payload).execute()
    _raise_if_response_error(resp, f"insert {table_name}")
    return resp.data  # list of inserted rows (if returning configured)

def _get_inserted_id(inserted_rows, preferred_names=("id","image_id","retina_id")):
    if not inserted_rows:
        return None
    row = inserted_rows[0]
    for n in preferred_names:
        if n in row:
            return row[n]
    # fallback to first column
    return next(iter(row.values()))

# ----- load local model directly -----
# ----- local model path -----
LOCAL_MODEL_PATH = os.path.join(MODEL_FOLDER, "retinova_model.h5")

# ----- lazy pipeline loader -----
pipeline = None

def get_pipeline():
    """
    Lazily initialize the model pipeline when first needed.
    This keeps /docs fast.
    """
    global pipeline

    if pipeline is not None:
        return pipeline

    if WrapperClass is None:
        raise RuntimeError(f"Could not import wrapper: {wrapper_import_error}")

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(f"Local model file missing: {LOCAL_MODEL_PATH}")

    log.info("Initializing wrapper with model: %s", LOCAL_MODEL_PATH)
    loaded = WrapperClass(model_path=LOCAL_MODEL_PATH)
    log.info("Wrapper initialized successfully.")
    pipeline = loaded
    return pipeline

@app.get("/hello")
async def hello():
    return {"msg": "hello"}

# ----- main endpoint -----
# ----- main endpoint -----
@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    user_email: str = Form(...),
    user_id: str = Form(None)
):
    # Basic MIME validation
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images allowed.")

    # Ensure pipeline available (lazy init)
    try:
        pipeline_instance = get_pipeline()
    except Exception as e:
        log.exception("Pipeline initialization failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model pipeline error: {e}")

    # Save input file locally
    ext = file.filename.split(".")[-1]
    local_input_name = f"{uuid.uuid4()}.{ext}"
    local_input_path = os.path.join(UPLOAD_FOLDER, local_input_name)
    content_bytes = await file.read()
    with open(local_input_path, "wb") as f:
        f.write(content_bytes)
    log.info("Saved input image to %s", local_input_path)

    try:
        # Run wrapper (must return dict described below)
        log.info("Running wrapper on %s", local_input_path)
        wrapper_result = pipeline_instance.run_with_path(local_input_path)
        log.debug("Wrapper result: %s", {k: type(v) for k, v in wrapper_result.items()})

        # Upload original image to user-images bucket (private)
        bucket_image_name = f"{uuid.uuid4()}.{ext}"
        _upload_bytes_to_bucket(
            BUCKET_USER_IMAGES,
            bucket_image_name,
            content_bytes,
            content_type=file.content_type,
        )
        main_signed = _signed_url_from_bucket(
            BUCKET_USER_IMAGES, bucket_image_name, expires_sec=24 * 3600
        )

        # Upload GradCAM and Heatmap if wrapper provided local paths
        gradcam_signed = None
        heatmap_signed = None

        # Accept multiple possible key names from wrapper
        grad_path = wrapper_result.get("gradcam_path") or wrapper_result.get(
            "gradcam_overlay_path"
        )
        heat_path = wrapper_result.get("heatmap_path") or wrapper_result.get(
            "heatmap_image_path"
        )

        # Normalize paths to absolute if necessary
        if grad_path and not os.path.isabs(grad_path):
            grad_path = os.path.join(os.getcwd(), grad_path)
        if heat_path and not os.path.isabs(heat_path):
            heat_path = os.path.join(os.getcwd(), heat_path)

        if grad_path and os.path.exists(grad_path):
            with open(grad_path, "rb") as gf:
                gbytes = gf.read()
            gname = f"{uuid.uuid4()}_overlay.png"
            _upload_bytes_to_bucket(
                BUCKET_OVERLAY, gname, gbytes, content_type="image/png"
            )
            gradcam_signed = _signed_url_from_bucket(
                BUCKET_OVERLAY, gname, expires_sec=24 * 3600
            )
            log.info("Uploaded gradcam -> %s", gname)
        else:
            log.info("No gradcam produced or file missing: %s", grad_path)

        if heat_path and os.path.exists(heat_path):
            with open(heat_path, "rb") as hf:
                hbytes = hf.read()
            hname = f"{uuid.uuid4()}_heatmap.png"
            _upload_bytes_to_bucket(
                BUCKET_HEATMAP, hname, hbytes, content_type="image/png"
            )
            heatmap_signed = _signed_url_from_bucket(
                BUCKET_HEATMAP, hname, expires_sec=24 * 3600
            )
            log.info("Uploaded heatmap -> %s", hname)
        else:
            log.info("No heatmap produced or file missing: %s", heat_path)

        # Insert retina_images row
        retina_payload = {
            "user_id": user_id,
            "user_email": user_email,
            "file_path": bucket_image_name,
            "image_url": main_signed,
            "gradcam_url": gradcam_signed,
            "heatmap_url": heatmap_signed,
        }
        inserted_retina = _insert_table("retina_images", retina_payload)
        image_id = _get_inserted_id(inserted_retina)
        log.info("Inserted retina_images row id=%s", image_id)

        # Insert prediction row
        prediction_payload = {
            "image_id": image_id,
            "user_id": user_id,
            "model_version": wrapper_result.get("model_version"),
            "predicted_condition": wrapper_result.get("prediction"),
            "confidence": wrapper_result.get("prediction_confidence"),
        }
        _insert_table("predictions", prediction_payload)

        # Insert results row
        results_payload = {
            "image_id": image_id,
            "user_id": user_id,
            "user_email": user_email,
            "final_diagnosis": wrapper_result.get("prediction"),
            "final_accuracy": wrapper_result.get("prediction_confidence"),
        }
        _insert_table("results", results_payload)

        # Insert MCQ responses if present
        mcq_list = wrapper_result.get("mcq_questions") or wrapper_result.get(
            "mcq_responses"
        ) or []
        for q in mcq_list:
            mcq_payload = {
                "user_id": user_id,
                "user_email": user_email,
                "question_text": q.get("question_text") or q.get("question"),
                "selected_option": q.get("selected_option") or q.get("user_answer"),
                "correct_option": q.get("correct_option"),
                "is_correct": q.get("is_correct", False),
            }
            _insert_table("mcq_responses", mcq_payload)

        # Build response
        return {
            "message": "Processed successfully",
            "image_id": image_id,
            "main_signed_url": main_signed,
            "gradcam_signed_url": gradcam_signed,
            "heatmap_signed_url": heatmap_signed,
            "prediction": wrapper_result.get("prediction"),
            "confidence": wrapper_result.get("prediction_confidence"),
            "mcq_count": len(mcq_list),
        }

    except Exception as exc:
        log.exception("Processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # remove uploaded input image from disk to save space
        try:
            if os.path.exists(local_input_path):
                os.remove(local_input_path)
        except Exception:
            pass