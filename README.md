# Chilli Fruit/Leaf Disease Detection - Full Stack Deep Learning Project

## Architecture
Frontend (React) -> Backend (FastAPI) -> TensorFlow/Keras model -> prediction JSON -> Frontend

This starter project uses transfer learning with MobileNetV2. The training script expects an image-folder dataset:
data/chilli/
  healthy/
  bacterial_spot/
  leaf_curl/
  leaf_spot/
  whitefly/
  yellowing/

You can change the class folders to match your dataset. The class names are read automatically.

## 1. Install
Python 3.10+ is recommended.

cd ml
pip install -r requirements.txt

For backend:
cd ../backend
pip install -r requirements.txt

For frontend:
cd ../frontend
npm install

## 2. Dataset
A small public chilli dataset with five classes is described in a peer-reviewed paper:
https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1367738/full
The paper identifies the source as the Kaggle "chili-plant-disease" dataset.

For PlantVillage pepper data (healthy + bacterial spot):
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

For a larger general plant-disease dataset:
https://huggingface.co/datasets/mohanty/PlantVillage

Do not put the complete dataset in this ZIP; datasets can be hundreds of MB/GB.

## 3. Train
Place the dataset under data/chilli/<class-name>/image.jpg.

python ml/train.py

This creates:
backend/models/chilli_disease.keras
backend/models/class_names.json

## 4. Start backend
cd backend
uvicorn app:app --reload --port 8000

Open http://127.0.0.1:8000/docs

## 5. Start frontend
cd frontend
npm run dev

The UI uploads one image to POST /predict and displays disease + confidence.

## Important
This is a classification demonstration, not a medical/agricultural diagnostic guarantee. Real field images can differ significantly from curated datasets.
