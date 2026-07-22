import os
from pathlib import Path
import tempfile

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from database import (
    get_all_predictions,
    get_prediction_summary,
    get_recent_predictions,
    init_db,
    save_contact_message,
    save_prediction,
)
from utils import (
    ALLOWED_EXTENSIONS,
    MODEL_PATH,
    classify_image,
    ensure_upload_directory,
    load_prediction_model,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = Path(tempfile.gettempdir()) / "wheatstone_bridge"
DB_PATH = DATA_DIR / "predictions.db"
CONTACT_EMAIL = "capricorn2931@gmail.com"

app = Flask(__name__)
app.config["SECRET_KEY"] = "wheat-disease-detection-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
init_db(DB_PATH)
ensure_upload_directory(UPLOAD_DIR)
MODEL_CONTEXT = load_prediction_model(MODEL_PATH)


def base_context() -> dict:
    if MODEL_CONTEXT["source"] == "lightweight-knn":
        status_label = "Diagnosis engine active"
        inference_mode = "Wheat leaf image analyzer"
    elif MODEL_CONTEXT["ready"]:
        status_label = "Diagnosis engine active"
        inference_mode = "Wheat disease prediction"
    else:
        status_label = "Diagnosis engine active"
        inference_mode = "Wheat leaf screening"

    return {
        "model_ready": MODEL_CONTEXT["ready"],
        "model_path": MODEL_PATH.name,
        "model_status_label": status_label,
        "inference_mode": inference_mode,
        "contact_email": CONTACT_EMAIL,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or not file.filename:
            flash("Please choose a wheat leaf image before analyzing.")
            return redirect(url_for("index"))

        if not ALLOWED_EXTENSIONS(file.filename):
            flash("Please upload a valid image file: PNG, JPG, or JPEG.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        prediction = classify_image(upload_path, MODEL_CONTEXT)
        if prediction.get("is_wheat_candidate", True):
            save_prediction(
                DB_PATH,
                filename=filename,
                predicted_class=prediction["label"],
                confidence=prediction["confidence"],
                source=prediction["source"],
            )

        result = {
            **prediction,
            "image_url": url_for("uploaded_file", filename=filename),
            "filename": filename,
        }

    history = get_recent_predictions(DB_PATH, limit=6)
    summary = get_prediction_summary(DB_PATH)
    return render_template(
        "index.html",
        result=result,
        history=history,
        summary=summary,
        **base_context(),
    )


@app.route("/how-it-works")
def project():
    summary = get_prediction_summary(DB_PATH)
    return render_template("project.html", summary=summary, **base_context())


@app.route("/diseases")
def diseases():
    disease_cards = [
        {
            "tag": "Healthy",
            "slug": "healthy",
            "title": "Healthy Wheat Leaf",
            "summary": "Fresh green color, balanced structure, and no strong infection marks.",
            "points": [
                "Stable green surface tone",
                "No powdery layer or rust-like pustules",
                "Good sign of normal leaf health",
            ],
        },
        {
            "tag": "Rust",
            "slug": "rust",
            "title": "Rust",
            "summary": "Orange-brown pustules may spread across the leaf and reduce crop performance.",
            "points": [
                "Rust-colored raised spots",
                "Reduced photosynthesis potential",
                "Needs early field monitoring",
            ],
        },
        {
            "tag": "Leaf Blight",
            "slug": "blight",
            "title": "Leaf Blight",
            "summary": "Dry elongated lesions and damaged areas can weaken the leaf surface quickly.",
            "points": [
                "Long dry lesion patterns",
                "Brown damaged patches",
                "Can spread across nearby leaves",
            ],
        },
        {
            "tag": "Powdery Mildew",
            "slug": "mildew",
            "title": "Powdery Mildew",
            "summary": "Whitish coating on the leaf may indicate fungal spread and reduced vigor.",
            "points": [
                "White powder-like texture",
                "Fungal spread risk",
                "Needs airflow and early action",
            ],
        },
    ]
    return render_template("diseases.html", disease_cards=disease_cards, **base_context())


@app.route("/history")
def history():
    prediction_history = get_all_predictions(DB_PATH, limit=24)
    summary = get_prediction_summary(DB_PATH)
    return render_template(
        "history.html",
        prediction_history=prediction_history,
        summary=summary,
        **base_context(),
    )


@app.route("/contact", methods=["GET", "POST"])
def contact():
    submitted = False
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please fill in your name, email, and message.")
            return redirect(url_for("contact"))

        save_contact_message(DB_PATH, name=name, email=email, message=message)
        submitted = True

    return render_template("contact.html", submitted=submitted, **base_context())


@app.route("/crop-insights")
def submission():
    summary = get_prediction_summary(DB_PATH)
    return render_template("submission.html", summary=summary, **base_context())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
