import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
import urllib.request
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR / "01_Model"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# Worker globals
_yunet = None
_sface = None

def download_sface():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not SFACE_PATH.exists():
        print(f"Downloading SFace model to {SFACE_PATH}...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
        print("Download complete.")

def init_worker(yunet_path, sface_path):
    global _yunet, _sface
    _yunet = cv2.FaceDetectorYN.create(
        model=str(yunet_path),
        config="",
        input_size=(320, 320),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )
    _sface = cv2.FaceRecognizerSF.create(
        model=str(sface_path),
        config="",
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )

def load_image_cv2(file_path):
    try:
        pil_img = Image.open(file_path).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)
        open_cv_image = np.array(pil_img)
        if open_cv_image.ndim == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    except Exception as e:
        return None

def process_image(row_dict):
    file_path = row_dict.get('location')
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        return []
        
    img = load_image_cv2(file_path)
    if img is None:
        return []
        
    h, w, _ = img.shape
    _yunet.setInputSize((w, h))
    
    _, faces = _yunet.detect(img)
    results = []
    
    if faces is not None:
        for face_idx, face in enumerate(faces):
            aligned_face = _sface.alignCrop(img, face)
            face_feature = _sface.feature(aligned_face)
            
            # encode aligned_face to jpg bytes to save IPC bandwidth
            _, buffer = cv2.imencode('.jpg', aligned_face)
            
            results.append({
                'embedding': face_feature[0],
                'face_img_bytes': buffer.tobytes(),
                'file_path': file_path,
                'face_idx': face_idx
            })
            
    return results

def main(catalog_path, profiles_dir, limit=None, workers=None):
    print("Initializing models...")
    download_sface()
    
    if not YUNET_PATH.exists():
        print("Error: YuNet model not found.")
        return
        
    print(f"Loading catalog {catalog_path}...")
    df = pd.read_csv(catalog_path)
    face_df = df[df['face_detected'] == True].copy()
    
    if limit:
        face_df = face_df.head(limit)
        print(f"Limiting to first {limit} photos.")
        
    print(f"Found {len(face_df)} photos to process.")
    
    rows = face_df.to_dict('records')
    
    embeddings = []
    metadata = []
    face_img_bytes_list = []
    
    num_workers = workers if workers else max(1, cpu_count() - 2)
    print(f"Extracting and embedding faces using {num_workers} workers...")
    
    # Process images in parallel
    with Pool(processes=num_workers, initializer=init_worker, initargs=(YUNET_PATH, SFACE_PATH)) as pool:
        for results in tqdm(pool.imap_unordered(process_image, rows), total=len(rows)):
            for res in results:
                embeddings.append(res['embedding'])
                face_img_bytes_list.append(res['face_img_bytes'])
                metadata.append((res['file_path'], res['face_idx']))

    if not embeddings:
        print("No faces could be extracted.")
        return
        
    print(f"\nSuccessfully extracted {len(embeddings)} individual faces.")
    
    # Cluster embeddings
    print("Clustering faces...")
    # Normalize embeddings for distance computation
    X = np.array(embeddings)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # DBSCAN parameters: eps (max distance between neighbors), min_samples (min cluster size)
    clustering = DBSCAN(eps=0.5, min_samples=3, metric='euclidean').fit(X_norm)
    labels = clustering.labels_
    
    out_dir = Path(profiles_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving clustered profiles to {out_dir}...")
    
    cluster_counts = {}
    for label in labels:
        if label != -1:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
            
    num_clusters = len(cluster_counts)
    print(f"Found {num_clusters} unique people profiles (ignoring unclassified faces).")
    
    for i, label in enumerate(labels):
        if label == -1:
            person_dir = out_dir / "Unclassified"
        else:
            person_dir = out_dir / f"Person_{label:03d}"
            
        person_dir.mkdir(parents=True, exist_ok=True)
        
        original_stem = Path(metadata[i][0]).stem
        face_idx = metadata[i][1]
        
        out_file = person_dir / f"{original_stem}_face{face_idx}.jpg"
        
        # Decode and save
        nparr = np.frombuffer(face_img_bytes_list[i], np.uint8)
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(str(out_file), img_decode)
        
    print("Done!")

if __name__ == "__main__":
    DEFAULT_CATALOG = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog_enriched.csv"
    DEFAULT_OUTDIR = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "profiles"

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG), help="Path to enriched catalog CSV")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Path to save profiles")
    parser.add_argument("--limit", type=int, default=None, help="Number of photos to process for testing")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers")
    args = parser.parse_args()
    
    main(args.catalog, args.outdir, args.limit, args.workers)
