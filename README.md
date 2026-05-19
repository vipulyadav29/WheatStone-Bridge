# WheatStone Bridge

WheatStone Bridge is an AI-driven wheat leaf disease detection web application built as a final year project.  
It helps users upload a wheat leaf image, detect the likely disease class, and understand crop-health conditions through a clean web interface.

## Project Highlights

- Wheat-only disease detection platform
- Flask-based backend with live image upload flow
- Trained lightweight machine learning classifier
- Disease classes: `Healthy`, `Rust`, `Leaf Blight`, `Powdery Mildew`
- Diagnosis history storage with SQLite support
- Product-style frontend for presentation and demo

## Tech Stack

- `Python`
- `Flask`
- `HTML`
- `CSS`
- `JavaScript`
- `SQLite`
- `Machine Learning`

## Folder Structure

```text
WheatStone-Bridge-GitHub/
  app.py
  database.py
  utils.py
  requirements.txt
  model/
  static/
  templates/
```

## How It Works

1. User uploads a wheat leaf image.
2. The system checks basic image quality.
3. Visual features are extracted from the image.
4. The trained classifier predicts the likely disease class.
5. The result is shown with confidence and short guidance.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Project Purpose

This project demonstrates how machine learning and web development can be combined to create a practical agriculture support platform for wheat disease screening and awareness.
