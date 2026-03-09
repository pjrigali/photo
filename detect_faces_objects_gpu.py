import os
import argparse
import pandas as pd
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from multiprocessing import Pool

from ultralytics import YOLO

# Allow loading of truncated images to prevent crashing on slightly corrupted JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CATALOG = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog.csv"
DEFAULT_OUTPUT = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog_enriched.csv"
DEFAULT_FACE_MODEL = SCRIPT_DIR / "01_Model" / "face_detection_yunet_2023mar.onnx"
DEFAULT_YOLO_MODEL = SCRIPT_DIR / "yolo11n.pt"

# Global worker variables for the CPU preprocessing pool
_face_detector = None
_tile_size = 640
_overlap = 120

def init_preproc_worker(face_model_path, tile_size, overlap):
    """
    Initialize CPU-bound models (YuNet) once per worker process.
    PyTorch and YOLO are NOT loaded here to avoid GPU OOM issues.
    """
    global _face_detector, _tile_size, _overlap
    _tile_size = tile_size
    _overlap = overlap
    
    # Initialize YuNet Face Detector
    if os.path.exists(face_model_path):
        _face_detector = cv2.FaceDetectorYN.create(
            model=str(face_model_path),
            config="",
            input_size=(320, 320), # Dummy initial size
            score_threshold=0.80,
            nms_threshold=0.3,
            top_k=5000
        )
    else:
        print(f"Warning: Face model not found at {face_model_path}. Face detection skipped.")


def tile_image(image: Image.Image, tile_size: int, overlap: int):
    """Splits an image into overlapping tiles."""
    img_width, img_height = image.size
    tiles = []
    stride = tile_size - overlap
    
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            left = x
            top = y
            right = min(x + tile_size, img_width)
            bottom = min(y + tile_size, img_height)

            # Ensure valid sizes for edge tiles
            if right - left < tile_size:
                left = max(0, img_width - tile_size)
            if bottom - top < tile_size:
                top = max(0, img_height - tile_size)
            
            tile = image.crop((left, top, right, bottom))
            tiles.append({'tile': tile, 'coords': (left, top)})
            
            if x + tile_size >= img_width:
                break
        if y + tile_size >= img_height:
            break
            
    return tiles


def preprocess_image(row_dict):
    """
    CPU Worker function:
    Loads image, applies EXIF rotation, detects faces via YuNet, and generates tiles.
    Returns the original row_dict + processed arrays for the main GPU thread.
    """
    global _face_detector, _tile_size, _overlap
    
    file_path = str(row_dict.get("location", ""))
    
    result = {
        "row_dict": row_dict,
        "face_detected": False,
        "number_of_faces": 0,
        "error": "",
        "tiles": [],
        "skip": False
    }
    
    if not os.path.exists(file_path):
        result["error"] = "File not found"
        result["skip"] = True
        return result
        
    try:
        # 1. Load Image with Pillow
        img = Image.open(file_path).convert('RGB')
        
        # 2. Apply Orientation from metadata
        rot = row_dict.get("rotation_needed", "")
        if pd.notna(rot) and rot != "" and str(rot) != "0":
            img = img.rotate(int(rot), expand=True)
            
        flip = str(row_dict.get("flip_needed", "False")).lower() == "true"
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        # 3. Face Detection using YuNet (CPU)
        if _face_detector is not None:
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            _face_detector.setInputSize((w, h))
            try:
                _, faces = _face_detector.detect(img_bgr)
                num_faces = len(faces) if faces is not None else 0
                result["face_detected"] = num_faces > 0
                result["number_of_faces"] = num_faces
            except Exception as e:
                result["error"] += f"Face detection failed: {e} "
        
        # 4. Generate Tiles for GPU Object Detection
        raw_tiles = tile_image(img, _tile_size, _overlap)
        
        # Convert to numpy/RGB to pass safely across process boundaries
        for t in raw_tiles:
            result["tiles"].append({
                'tile_np': np.array(t['tile']),
                'coords': t['coords']
            })
            
    except Exception as e:
        result["error"] += f"Image processing error: {e}"
        result["skip"] = True
        
    return result


def apply_nms(detections, iou_threshold=0.5):
    """Applies Non-Maximum Suppression to filter duplicate detections across tiles."""
    if len(detections) == 0:
        return []

    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5]

    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    classes_tensor = torch.from_numpy(classes)

    unique_classes = torch.unique(classes_tensor)
    keep_indices = []

    for cls in unique_classes:
        cls_indices = (classes_tensor == cls).nonzero(as_tuple=False).flatten()
        cls_boxes = boxes_tensor[cls_indices]
        cls_scores = scores_tensor[cls_indices]
        
        keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.append(cls_indices[keep])
    
    if not keep_indices:
        return []
        
    keep_indices = torch.cat(keep_indices, dim=0)
    return detections[keep_indices.cpu().numpy()]


def main():
    parser = argparse.ArgumentParser(description="GPU-Optimized Detect Faces and Objects")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG), help="Path to input photo catalog CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output catalog path")
    parser.add_argument("--face-model", type=str, default=str(DEFAULT_FACE_MODEL), help="Path to YuNet ONNX model")
    parser.add_argument("--yolo-model", type=str, default=str(DEFAULT_YOLO_MODEL), help="Path to YOLO model")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent CPU loading processes")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for object detection")
    parser.add_argument("--overlap", type=int, default=120, help="Tile overlap for object detection")
    parser.add_argument("--batch-limit", type=int, default=0, help="Limit number of photos to process (for testing)")
    
    args = parser.parse_args()
    
    catalog_path = Path(args.catalog).resolve()
    output_path = Path(args.output).resolve()
    
    if not catalog_path.exists():
        print(f"Error: Catalog not found at {catalog_path}")
        return
        
    print(f"[{catalog_path.name}] Loading catalog...")
    df = pd.read_csv(catalog_path, encoding="utf-8-sig")
    
    # Ensure orientation metadata is present
    if "rotation_needed" not in df.columns:
        df["rotation_needed"] = ""
    if "flip_needed" not in df.columns:
        df["flip_needed"] = ""
        
    # Checkpoint: Find already processed locations
    processed_locations = set()
    if output_path.exists():
        try:
            existing_out_df = pd.read_csv(output_path, encoding="utf-8-sig", low_memory=False)
            if "location" in existing_out_df.columns:
                processed_locations = set(existing_out_df["location"].dropna().astype(str))
            print(f"Found {len(processed_locations)} already processed images in existing output.")
        except Exception as e:
            print(f"Failed to read existing output catalog for checkpointing: {e}")
            
    records = df.to_dict("records")
    
    # Filter out already processed images
    filtered_records = [r for r in records if str(r.get("location", "")) not in processed_locations]
    
    if args.batch_limit > 0:
        filtered_records = filtered_records[:args.batch_limit]
        print(f"Limiting to {args.batch_limit} remaining records for testing...")
        
    total = len(filtered_records)
    if total == 0:
        print("All images have already been processed.")
        return
        
    # Setup GPU Infrastructure
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n============================================================")
    print(f"  GPU Pipeline Initialized")
    print(f"  Device:         {device}")
    print(f"  CPU Workers:    {args.workers}")
    print(f"  Target Images:  {total}")
    print(f"============================================================\n")
    
    # Load YOLO natively on the main process
    yolo_model = YOLO(str(args.yolo_model))
    yolo_model.to(device)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_headers = not output_path.exists()
    
    columns_to_keep = [
        "location", "folder_name", "file_name", "datetime_original",
        "orientation_label", "rotation_needed", "flip_needed",
        "face_detected", "number_of_faces", 
        "objects_detected", "objects_detected_set", "error"
    ]
    
    # Execute Multiprocessing Pool for CPU Preprocessing tasks
    pool = Pool(
        processes=args.workers,
        initializer=init_preproc_worker,
        initargs=(args.face_model, args.tile_size, args.overlap)
    )
    
    faces_detected_count = 0
    max_gpu_batch = 16 # Adjust this up or down to pace GPU VRAM
    
    with open(output_path, 'a', newline='', encoding='utf-8-sig') as f:
        # pool.imap streams results back as soon as they finish
        for i, res in enumerate(pool.imap(preprocess_image, filtered_records), 1):
            
            row_dict = res["row_dict"]
            row_dict["face_detected"] = res["face_detected"]
            row_dict["number_of_faces"] = res["number_of_faces"]
            row_dict["objects_detected"] = ""
            row_dict["error"] = res["error"]
            
            if res["face_detected"]:
                faces_detected_count += 1
                
            tiles = res["tiles"]
            final_objects = []
            
            if not res["skip"] and tiles:
                # Compile all tiles for this image into a single PyTorch batch!
                # Reconstruct numpy back to PIL or pass numpy array sequence. YOLO handles sequence of np arrays perfectly.
                tile_images = [t['tile_np'] for t in tiles]
                
                # Predict in high-throughput chunks on the GPU to not blow out VRAM
                all_objects = []
                for b_idx in range(0, len(tile_images), max_gpu_batch):
                    batch_imgs = tile_images[b_idx : b_idx + max_gpu_batch]
                    
                    # Predict batch
                    yolo_results = yolo_model.predict(source=batch_imgs, verbose=False, conf=0.50, device=device)
                    
                    for sub_idx, yolo_res in enumerate(yolo_results):
                        original_idx = b_idx + sub_idx
                        left, top = tiles[original_idx]['coords']
                        boxes = yolo_res.boxes.xyxy.cpu().numpy()
                        confs = yolo_res.boxes.conf.cpu().numpy()
                        clss = yolo_res.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            orig_x1, orig_y1 = box[0] + left, box[1] + top
                            orig_x2, orig_y2 = box[2] + left, box[3] + top
                            all_objects.append([orig_x1, orig_y1, orig_x2, orig_y2, conf, cls])
                        
                # Apply NMS for objects across all tiles
                if all_objects:
                    all_objects_np = np.array(all_objects)
                    nms_objects = apply_nms(all_objects_np, iou_threshold=0.5)
                    for d in nms_objects:
                        cls_idx = int(d[5])
                        final_objects.append(yolo_model.names[cls_idx])
                        
                row_dict["objects_detected"] = ", ".join(final_objects)
                
                # Derive unique objects and sort them alphabetically
                unique_objects = sorted(list(set(final_objects)))
                row_dict["objects_detected_set"] = ", ".join(unique_objects)
                
            # Filter output to only relevant join fields and detection results
            res_filtered = {col: row_dict.get(col, "") for col in columns_to_keep}
            row_df = pd.DataFrame([res_filtered])
            row_df = row_df[columns_to_keep]
            
            # Save checkpoint
            row_df.to_csv(f, header=write_headers, index=False)
            write_headers = False
            f.flush()
            
            if i % 10 == 0 or i == total:
                print(f"  -> Processed {i}/{total} images...")
                
    pool.close()
    pool.join()
    
    print(f"\n============================================================")
    print(f"  GPU Detection Complete")
    print(f"============================================================")
    print(f"  Total Processed This Run:     {total}")
    print(f"  New Faces Detected This Run:  {faces_detected_count}")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
