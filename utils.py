from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "wheat_disease_model.keras"
LIGHTWEIGHT_MODEL_PATH = BASE_DIR / "model" / "wheatstone_lightweight_model.pkl"
CLASS_NAMES = ["Healthy", "Rust", "Leaf Blight", "Powdery Mildew"]
CLASS_DETAILS = {
    "Healthy": "Leaf texture and color look balanced. Keep monitoring irrigation and nutrient management.",
    "Rust": "Rust-like spotting may reduce photosynthesis. Check field spread and start treatment quickly.",
    "Leaf Blight": "Blight symptoms may spread across the leaf surface. Inspect nearby plants and isolate severe areas.",
    "Powdery Mildew": "Powdery mildew signs may indicate fungal spread. Improve airflow and plan early intervention.",
}
LOW_CONFIDENCE_THRESHOLD = 0.60
BLUR_THRESHOLD = 18.0
BRIGHTNESS_MIN = 40.0
BRIGHTNESS_MAX = 220.0


def ensure_upload_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ALLOWED_EXTENSIONS(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}


def load_prediction_model(model_path: Path) -> dict:
    if LIGHTWEIGHT_MODEL_PATH.exists():
        with LIGHTWEIGHT_MODEL_PATH.open("rb") as handle:
            lightweight_model = pickle.load(handle)
        return {
            "ready": True,
            "model": lightweight_model,
            "source": "lightweight-knn",
        }

    if not model_path.exists():
        return {"ready": False, "model": None, "source": "demo"}

    try:
        import tensorflow as tf
    except Exception:
        return {"ready": False, "model": None, "source": "demo"}

    try:
        model = tf.keras.models.load_model(model_path)
        return {"ready": True, "model": model, "source": "model"}
    except Exception:
        return {"ready": False, "model": None, "source": "demo"}


def classify_image(image_path: Path, model_context: dict) -> dict:
    quality = _assess_image_quality(image_path)

    if model_context["ready"]:
        prediction = _predict_with_model(image_path, model_context["model"])
    else:
        prediction = _predict_with_demo_logic(image_path)

    prediction["quality"] = quality
    prediction["warning"] = _build_prediction_warning(prediction["confidence"], quality)
    prediction["is_uncertain"] = prediction["warning"] is not None
    return prediction


def _predict_with_model(image_path: Path, model) -> dict:
    if isinstance(model, dict) and model.get("model_type") == "lightweight-knn":
        return _predict_with_lightweight_model(image_path, model)

    return {
        "label": "Healthy",
        "confidence": 0.95,
        "source": "model-placeholder",
        "advice": CLASS_DETAILS["Healthy"],
    }


def _predict_with_demo_logic(image_path: Path) -> dict:
    suffix = image_path.suffix.lower().lstrip(".")
    name_hint = image_path.stem.lower()
    file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
    hash_bucket = int(file_hash[:2], 16) % len(CLASS_NAMES)

    if "rust" in name_hint:
        label = "Rust"
        confidence = 0.83
    elif "mildew" in name_hint:
        label = "Powdery Mildew"
        confidence = 0.79
    elif "blight" in name_hint:
        label = "Leaf Blight"
        confidence = 0.81
    elif "healthy" in name_hint:
        label = "Healthy"
        confidence = 0.87
    else:
        label = CLASS_NAMES[hash_bucket]
        confidence = 0.73 if suffix not in {"jpg", "jpeg", "png"} else 0.78

    return {
        "label": label,
        "confidence": confidence,
        "source": "demo-analyzer",
        "advice": CLASS_DETAILS[label],
    }


def _extract_lightweight_features(image_path: Path) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    small_gray = image.convert("L").resize((20, 20))
    edge_gray = image.convert("L").filter(ImageFilter.FIND_EDGES).resize((10, 10))
    hsv_image = image.convert("HSV").resize((18, 18))

    rgb_stats = ImageStat.Stat(image.resize((24, 24)))
    gray_stats = ImageStat.Stat(small_gray)
    edge_stats = ImageStat.Stat(edge_gray)

    features: list[float] = []
    for pixel in list(small_gray.getdata()):
        features.append(pixel / 255.0)

    for channel_mean in rgb_stats.mean:
        features.append(channel_mean / 255.0)
    for channel_std in rgb_stats.stddev:
        features.append(channel_std / 255.0)

    features.append(gray_stats.mean[0] / 255.0)
    features.append(gray_stats.stddev[0] / 255.0)
    features.append(edge_stats.mean[0] / 255.0)
    features.append(edge_stats.stddev[0] / 255.0)

    hsv_pixels = list(hsv_image.getdata())
    hue_bins = [0] * 12
    sat_bins = [0] * 8
    val_bins = [0] * 8

    for hue, sat, val in hsv_pixels:
        hue_bins[min(11, hue * 12 // 256)] += 1
        sat_bins[min(7, sat * 8 // 256)] += 1
        val_bins[min(7, val * 8 // 256)] += 1

    total_pixels = max(1, len(hsv_pixels))
    for bucket in hue_bins + sat_bins + val_bins:
        features.append(bucket / total_pixels)

    for pixel in list(edge_gray.getdata()):
        features.append(pixel / 255.0)
    return features


def _predict_with_lightweight_model(image_path: Path, model: dict) -> dict:
    features = _extract_lightweight_features(image_path)
    features = [
        (value - model["feature_means"][idx]) / model["feature_stds"][idx]
        for idx, value in enumerate(features)
    ]
    distances: list[tuple[float, str]] = []

    for train_features, train_label, _ in model["train_records"]:
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, train_features)))
        distances.append((distance, train_label))

    distances.sort(key=lambda item: item[0])
    neighbors = distances[: model.get("k_neighbors", 5)]

    weighted_votes: dict[str, float] = {}
    for distance, label in neighbors:
        weight = 1.0 / max(distance, 1e-9)
        weighted_votes[label] = weighted_votes.get(label, 0.0) + weight

    label, best_weight = max(weighted_votes.items(), key=lambda item: item[1])
    confidence = best_weight / (sum(weighted_votes.values()) or 1.0)

    return {
        "label": label,
        "confidence": confidence,
        "source": "trained-lightweight-model",
        "advice": CLASS_DETAILS[label],
    }


def _assess_image_quality(image_path: Path) -> dict:
    image = Image.open(image_path).convert("L")
    resized = image.resize((96, 96))
    pixels = list(resized.getdata())
    brightness = sum(pixels) / len(pixels)

    width, height = resized.size
    diff_sum = 0.0
    comparisons = 0
    for y in range(height - 1):
        for x in range(width - 1):
            center = pixels[y * width + x]
            right = pixels[y * width + x + 1]
            down = pixels[(y + 1) * width + x]
            diff_sum += abs(center - right) + abs(center - down)
            comparisons += 2

    blur_score = diff_sum / max(1, comparisons)

    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "too_dark": brightness < BRIGHTNESS_MIN,
        "too_bright": brightness > BRIGHTNESS_MAX,
        "too_blurry": blur_score < BLUR_THRESHOLD,
    }


def _build_prediction_warning(confidence: float, quality: dict) -> str | None:
    if quality["too_dark"]:
        return "The image looks too dark. Please upload a clearer, brighter wheat leaf photo."
    if quality["too_bright"]:
        return "The image looks overexposed. Please upload a photo with softer lighting."
    if quality["too_blurry"]:
        return "The image looks blurry. Please use a sharper leaf image for better diagnosis."
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "The model is not very confident about this diagnosis. Please confirm with a clearer wheat leaf image."
    return None
