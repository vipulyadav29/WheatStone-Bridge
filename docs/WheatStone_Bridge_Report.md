# WheatStone Bridge Project Report

## Overview

WheatStone Bridge is an AI-driven wheat leaf disease detection web application built as a final year project. It allows users to upload wheat leaf images and receive prediction output for the most likely disease class.

## Main Features

- Wheat-only disease detection workflow
- Image upload and result prediction
- Disease library and crop-health awareness
- Prediction history storage
- Contact and support interface
- Flask-based live web application

## Technologies Used

- Python
- Flask
- HTML, CSS, JavaScript
- SQLite
- Machine Learning classifier

## Machine Learning Summary

The project uses a lightweight trained weighted k-nearest neighbours classifier. It works on handcrafted image features such as grayscale intensity, RGB statistics, edge features, and HSV distribution. The current model predicts among four classes:

- Healthy
- Rust
- Leaf Blight
- Powdery Mildew

## Conclusion

WheatStone Bridge demonstrates how machine learning and web application development can be combined in the agriculture domain to build a practical crop-health screening platform.
