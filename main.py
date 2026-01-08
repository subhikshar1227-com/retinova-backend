import os
import uuid
import logging
import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import Body

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
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ----- helper wrappers for supabase responses -----
def _raise_if_response_error(resp, context: str = ""):
    if getattr(resp, "error", None):
        raise RuntimeError(f"{context} error: {resp.error}")

def _signed_url_from_bucket(bucket: str, path: str, expires_sec: int = 3600):
    resp = supabase.storage.from_(bucket).create_signed_url(path, expires_sec)
    _raise_if_response_error(resp, f"create_signed_url({bucket}/{path})")

    # ✅ storage3 returns dict, NOT .data
    if isinstance(resp, dict):
        return resp.get("signedURL")
 
    return None

def _upload_bytes_to_bucket(bucket: str, path: str, b: bytes, content_type: str = "image/png"):
    resp = supabase.storage.from_(bucket).upload(path, b, {"content-type": content_type})
    _raise_if_response_error(resp, f"upload({bucket}/{path})")
    return True

def _insert_table(table_name: str, payload: dict):
    resp = supabase.table(table_name).insert(payload).execute()
    _raise_if_response_error(resp, f"insert {table_name}")
    return resp.data  # list of inserted rows (if returning configured)

def _get_inserted_id(inserted_rows, preferred_names=("id", "image_id", "retina_id")):
    if not inserted_rows:
        return None
    row = inserted_rows[0]
    for n in preferred_names:
        if n in row:
            return row[n]
    # fallback to first column
    return next(iter(row.values()))

def _file_url_to_path(url: str | None):
    """
    Convert file:///C:/... style URL from the wrapper JSON
    into a normal Windows path C:\\... so os.path.exists works.
    """
    if not url:
        return None

    # file:///C:/Users/...  (3 slashes)
    if url.startswith("file:///"):
        path = url[len("file:///"):]
    # file://C:/Users/...   (2 slashes)
    elif url.startswith("file://"):
        path = url[len("file://"):]
    else:
        # already a path, just use it
        path = url

    # Normalize slashes for Windows
    return os.path.normpath(path)


# ----- model / pipeline (lazy load) -----
LOCAL_MODEL_PATH = os.path.join(MODEL_FOLDER, "retinova_model.h5")
pipeline = None

def get_pipeline():
    """
    Lazily initialize the model pipeline when first needed.
    For API use, we override ask_risk_questions to avoid terminal input().
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

    # --- OVERRIDE: no interactive MCQs in API ---
    def noninteractive_ask(condition, base_conf):
        # skip asking questions, just keep base confidence as final
        return float(base_conf), {}

    loaded.ask_risk_questions = noninteractive_ask
    # --------------------------------------------

    log.info("Wrapper initialized successfully (non-interactive).")
    pipeline = loaded
    return pipeline

# ----- health check -----
@app.get("/ping")
def ping():
    return {"status": "ok"}
from fastapi import Body

@app.post("/auth/register")
def register_user(
    user_id: str = Body(...),
    user_email: str = Body(...)
):
    """
    Creates a simple user record (no password).
    Frontend must reuse this same user_id everywhere.
    """

    payload = {
        "user_id": user_id,
        "user_email": user_email,
    }

    try:
        resp = supabase.table("users").insert(payload).execute()
        _raise_if_response_error(resp, "insert users")
    except Exception as e:
        log.exception("User registration failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail="User already exists or invalid payload"
        )

    return {
        "message": "User registered",
        "user_id": user_id,
        "user_email": user_email
    }
@app.post("/auth/login")
def login_user(
    user_id: str = Body(None),
    user_email: str = Body(None)
):
    """
    Login = existence check only.
    """

    if not user_id and not user_email:
        raise HTTPException(
            status_code=400,
            detail="Provide user_id or user_email"
        )

    query = supabase.table("users").select("*")

    if user_id:
        query = query.eq("user_id", user_id)
    else:
        query = query.eq("user_email", user_email)

    resp = query.limit(1).execute()
    _raise_if_response_error(resp, "select users")

    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")

    user = rows[0]

    return {
        "message": "Login success",
        "user_id": user["user_id"],
        "user_email": user["user_email"]
    }
# ----- main endpoint -----
@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    user_email: str = Form(...),
    user_id: str = Form(None),
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
    ext = (file.filename or "image").split(".")[-1]
    local_input_name = f"{uuid.uuid4()}.{ext}"
    local_input_path = os.path.join(UPLOAD_FOLDER, local_input_name)
    content_bytes = await file.read()
    with open(local_input_path, "wb") as f:
        f.write(content_bytes)
    log.info("Saved input image to %s", local_input_path)

    try:
        # Run wrapper (must return dict)
        log.info("Running wrapper on %s", local_input_path)
        wrapper_result = pipeline_instance.run_with_path(local_input_path)
        log.debug("Wrapper result keys: %s", list(wrapper_result.keys()))

        prediction = (
    wrapper_result.get("final_label")
    or wrapper_result.get("predicted_condition")
    or wrapper_result.get("prediction")
)

        confidence = (
    wrapper_result.get("final_confidence")
    or wrapper_result.get("base_confidence")
    or wrapper_result.get("prediction_confidence")
)

        # Upload original image to user-images bucket (private)
        bucket_image_name = f"{uuid.uuid4()}.{ext}"
        _upload_bytes_to_bucket(
            BUCKET_USER_IMAGES, bucket_image_name, content_bytes, content_type=file.content_type
        )
        main_signed = _signed_url_from_bucket(
            BUCKET_USER_IMAGES, bucket_image_name, expires_sec=24 * 3600
        )

        # Upload GradCAM and Heatmap if wrapper provided local paths
        grad_link = wrapper_result.get("gradcam_overlay") or wrapper_result.get("gradcam_overlay_path")
        heat_link = wrapper_result.get("gradcam_heatmap") or wrapper_result.get("gradcam_heatmap_path")

        gradcam_signed = None
        heatmap_signed = None

        grad_path = _file_url_to_path(grad_link)
        heat_path = _file_url_to_path(heat_link)

        log.info("DEBUG grad_link=%s heat_link=%s", grad_link, heat_link)
        log.info("DEBUG grad_path=%s heat_path=%s", grad_path, heat_path)

        if grad_path and not os.path.isabs(grad_path):
            grad_path = os.path.join(os.getcwd(), grad_path)
        if heat_path and not os.path.isabs(heat_path):
            heat_path = os.path.join(os.getcwd(), heat_path)

        if grad_path and os.path.exists(grad_path):
            with open(grad_path, "rb") as gf:
                gbytes = gf.read()
            gname = f"{uuid.uuid4()}_overlay.png"
            _upload_bytes_to_bucket(BUCKET_OVERLAY, gname, gbytes, content_type="image/png")
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
            _upload_bytes_to_bucket(BUCKET_HEATMAP, hname, hbytes, content_type="image/png")
            heatmap_signed = _signed_url_from_bucket(
                BUCKET_HEATMAP, hname, expires_sec=24 * 3600
            )
            log.info("Uploaded heatmap -> %s", hname)
        else:
            log.info("No heatmap produced or file missing: %s", heat_path)

# ---------------- JSON RESULTS BUILD ----------------
        import json

        json_payload = wrapper_result.copy()

        # Replace local paths with Supabase URLs
        json_payload["image_url"] = main_signed
        json_payload["gradcam_url"] = gradcam_signed
        json_payload["heatmap_url"] = heatmap_signed

        # Remove local-only paths (VERY IMPORTANT)
        for k in [
            "input_path",
            "gradcam_overlay_path",
            "gradcam_heatmap_path",
        ]:
            json_payload.pop(k, None)

        json_bytes = json.dumps(json_payload, indent=2).encode("utf-8")
        # ----------------------------------------------------


        # ---------------- JSON RESULTS UPLOAD ----------------
        json_signed = None
        jname = f"{uuid.uuid4()}_results.json"

        _upload_bytes_to_bucket(
            "results",                # bucket name
            jname,
            json_bytes,
            content_type="application/octet-stream" # ✅ FIXED MIME TYPE
        )

        json_signed = _signed_url_from_bucket(
            "results",
            jname,
            expires_sec=24 * 3600
        )

        log.info("Uploaded JSON results -> %s", jname)
        # ----------------------------------------------------

        log.info("FINAL URL CHECK -> image=%s grad=%s heat=%s",
         main_signed, gradcam_signed, heatmap_signed)

        # Insert retina_images row
        retina_payload = {
            "user_id":user_id,
            "user_email": user_email,
            "file_path":bucket_image_name,
            "image_url": main_signed,
            "gradcam_url":gradcam_signed,
            "heatmap_url":heatmap_signed,
            
    # uploaded_at will be set automatically by default now()
}
        inserted_retina = _insert_table("retina_images", retina_payload)
        image_id = _get_inserted_id(inserted_retina)
        log.info("Inserted retina_images row id=%s", image_id)
        log.info("DEBUG retina row: user_email=%s, image_url=%s, user_id=%s, gradcam_url=%s, heatmap_url=%s",
         user_email, main_signed,user_id,gradcam_signed,heatmap_signed)
        log.info("DEBUG retina_payload: %s",retina_payload)
        log.info("DEBUG grad_link=%s heat_link=%s", grad_link, heat_link)
        log.info("DEBUG grad_path=%s heat_path=%s", grad_path, heat_path)

        prediction_payload = {
    "image_id": image_id,
    "user_id": user_id,
    "disease": prediction,
    "probability": confidence,          # legacy field
    "base_confidence": confidence,      # CNN only
    "model_confidence": confidence,     # CNN only
    "final_confidence": confidence,     # same before MCQ
    "combined_confidence": confidence   # alias of final
}
        _insert_table("predictions", prediction_payload)

        results_payload = {
    "image_id": image_id,
    "user_id": user_id,
    "user_email": user_email,

    "final_accuracy": confidence,
    "diagnosis_summary": prediction,
    "json_report_url": json_signed,

    "base_confidence": confidence,
    "model_confidence": confidence,
    "final_confidence": confidence,
    "combined_confidence": confidence
}
        _insert_table("results", results_payload)

        # Insert MCQ responses if present
        mcq_list=[]

        # Build response
        return {
            "message": "Processed successfully",
            "image_id": image_id,
            "main_signed_url": main_signed,
            "gradcam_signed_url": gradcam_signed,
            "heatmap_signed_url": heatmap_signed,
            "prediction": prediction,
            "confidence": confidence,
            "mcq_count": len(mcq_list),
        }

    except Exception as exc:
        log.exception("Processing failed: %s", exc) 
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        try:
            if os.path.exists(local_input_path):
                os.remove(local_input_path)
        except Exception:
            pass
# ================= MCQ — FETCH QUESTIONS =================
@app.get("/mcq/questions/{image_id}")
def get_mcq_questions(image_id: str):
    """
    Returns the MCQ questions for the disease predicted for this image.
    """

    # 1) Look up prediction
    pred_row = (
        supabase.table("predictions")
        .select("*")
        .eq("image_id", image_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not pred_row.data:
        raise HTTPException(status_code=404, detail="No prediction found for this image")

    disease = pred_row.data[0].get("disease")

    # 2) Pull question bank
    pipeline_instance = get_pipeline()
    risk_bank = getattr(pipeline_instance, "risk_questions", {})

    disease_key = disease.strip().lower()

    questions = (
        risk_bank.get(disease_key)
        or risk_bank.get(disease)
        or []
    )

    # ✅ DEBUG LOG
    log.info(
        "MCQ FETCH → image_id=%s disease=%s questions=%d",
        image_id, disease, len(questions)
    )

    return {
        "image_id": image_id,
        "disease": disease,
        "question_count": len(questions),
        "questions": [
            {
                "id": i,
                "question": q,
                "choices": opts
            }
            for i, (q, opts) in enumerate(questions)
        ]
    }
@app.get("/mcq_questions/{image_id}")
def get_mcq_questions_alias(image_id: str):
    return get_mcq_questions(image_id)

@app.post("/mcq/submit")
def submit_mcq_answers(payload: dict = Body(...)):
    image_id = payload.get("image_id")
    user_id = payload.get("user_id")
    user_email = payload.get("user_email")
    answers = payload.get("answers", [])

    log.info(
        "MCQ PAYLOAD RECEIVED → image_id=%s answers=%s",
        image_id,
        answers
    )

    if not image_id or not answers:
        raise HTTPException(
            status_code=400,
            detail="image_id and answers[] are required"
        )

    # ---- fetch latest prediction for base confidence ----
    pred = (
        supabase.table("predictions")
        .select("*")
        .eq("image_id", image_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not pred.data:
        raise HTTPException(
            status_code=404,
            detail="No prediction found for image_id"
        )

    # ✅ SAFE BASE CONFIDENCE FALLBACK
    base_conf = float(
        pred.data[0].get("base_confidence")
        or pred.data[0].get("model_confidence")
        or pred.data[0].get("probability")
        or pred.data[0].get("confidence")
        or 0
    )

    # 🔐 normalize confidence if stored as percentage
    if base_conf > 1:
        base_conf = base_conf / 100

    final_conf = base_conf

    log.info(
        "MCQ SUBMIT → image_id=%s base_conf=%.4f final_conf=%.4f answers=%d",
        image_id, base_conf, final_conf, len(answers)
    )

    # ---- confidence bump rule ----
    CONF_BUMPS = {
        0: 0.00,
        1: 0.05,
        2: 0.10,
        3: 0.20
    }

    rows_inserted = 0

    for idx, ans in enumerate(answers):
        choice_idx = ans.get("selected_option_index", 0)
        choice_text = ans.get("choice_text") or ans.get("choice") or ""

        bump = CONF_BUMPS.get(choice_idx, 0.0)
        final_conf = min(1.0, final_conf + bump)

        row = {
            "image_id": image_id,
            "user_id": user_id,
            "user_email": user_email,
            "question_index": idx,
            "selected_option_index": choice_idx,
            "severity_level": choice_text,
            "adjusted_confidence": final_conf
        }

        supabase.table("mcq_responses").insert(row).execute()
        rows_inserted += 1

    # ---- update predictions ----
    supabase.table("predictions").update({
        "final_confidence": final_conf,
        "combined_confidence": final_conf
    }).eq("image_id", image_id).execute()

    # ---- update results ----
    supabase.table("results").update({
        "final_confidence": final_conf,
        "combined_confidence": final_conf,
        "final_accuracy": final_conf
    }).eq("image_id", image_id).execute()

    final_conf = round(final_conf, 4)

    return {
        "message": "MCQ responses stored successfully",
        "answers_saved": rows_inserted,
        "base_confidence": base_conf,
        "final_confidence": final_conf
    }
@app.post("/mcq_submit")
def submit_mcq_answers_alias(payload: dict = Body(...)):
    return submit_mcq_answers(payload)