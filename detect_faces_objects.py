import os
import argparse
import pandas as pd
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from multiprocessing import Pool, cpu_count

from ultralytics import YOLO

# Allow loading of truncated images to prevent crashing on slightly corrupted JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CATALOG = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog.csv"
DEFAULT_OUTPUT = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog_enriched.csv"
DEFAULT_FACE_MODEL = SCRIPT_DIR / "01_Model" / "face_detection_yunet_2023mar.onnx"
DEFAULT_YOLO_MODEL = SCRIPT_DIR / "yolo11n.pt"

# Global worker variables
_face_detector = None
_yolo_model = None
_tile_size = 640
_overlap = 120

def init_worker(face_model_path, yolo_model_path, tile_size, overlap):
    """
    Initialize models once per worker process to avoid pickling issues
    and redundant loading.
    """
    global _face_detector, _yolo_model, _tile_size, _overlap
    _tile_size = tile_size
    _overlap = overlap
    
    # Initialize YuNet Face Detector
    # Input size will be set dynamically per image
    if os.path.exists(face_model_path):
        _face_detector = cv2.FaceDetectorYN.create(
            model=str(face_model_path),
            config="",
            input_size=(320, 320), # Dummy initial size
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
    else:
        print(f"Warning: Face model not found at {face_model_path}. Face detection will be skipped.")
    
    # Initialize YOLO Model
    if os.path.exists(yolo_model_path):
        _yolo_model = YOLO(str(yolo_model_path))
    else:
        # Ultralytics can auto-download if just the name is provided, 
        # but since we expect a local file, we pass the path. It may download if path fails.
        _yolo_model = YOLO(str(yolo_model_path))


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


def process_image(row_dict):
    """Worker function to process a single image row."""
    global _face_detector, _yolo_model, _tile_size, _overlap
    
    file_path = str(row_dict.get("location", ""))
    
    # Defaults in case of skip/error
    row_dict["face_detected"] = False
    row_dict["number_of_faces"] = 0
    row_dict["objects_detected"] = ""
    row_dict["error"] = ""
    
    if not os.path.exists(file_path):
        row_dict["error"] = "File not found"
        print(f"File not found: {file_path}")
        return row_dict
        
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
            
        # 3. Face Detection using YuNet
        if _face_detector is not None:
            # Convert RGB PIL Image to BGR OpenCV Image
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            _face_detector.setInputSize((w, h))
            _face_detector.setScoreThreshold(0.80)
            try:
                _, faces = _face_detector.detect(img_bgr)
                num_faces = len(faces) if faces is not None else 0
                row_dict["face_detected"] = num_faces > 0
                row_dict["number_of_faces"] = num_faces
            except Exception as e:
                print(f"Face detection failed for {file_path}: {e}")
                row_dict["error"] = f"Face detection failed: {e}"
        
        # 4. Object Detection with Tiling
        if _yolo_model is not None:
            tiles = tile_image(img, _tile_size, _overlap)
            
            all_objects = []
            
            for tile_info in tiles:
                tile_img = tile_info['tile']
                left, top = tile_info['coords']
                
                # Predict on tile
                results_list = _yolo_model.predict(source=tile_img, verbose=False, conf=0.50)
                for results in results_list:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = box
                        # Map to original image coordinates
                        orig_x1, orig_y1 = x1 + left, y1 + top
                        orig_x2, orig_y2 = x2 + left, y2 + top
                        all_objects.append([orig_x1, orig_y1, orig_x2, orig_y2, conf, cls])
                        
            # Apply NMS for objects
            final_objects = []
            if all_objects:
                all_objects_np = np.array(all_objects)
                nms_objects = apply_nms(all_objects_np, iou_threshold=0.5)
                
                # Extract Class Name
                for d in nms_objects:
                    cls_idx = int(d[5])
                    final_objects.append(_yolo_model.names[cls_idx])
                    
            # Store objects as a comma-separated string
            row_dict["objects_detected"] = ", ".join(final_objects)
            
            # Derive unique objects and sort them alphabetically
            unique_objects = sorted(list(set(final_objects)))
            row_dict["objects_detected_set"] = ", ".join(unique_objects)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        row_dict["error"] = f"Image processing error: {e}"
        
    return row_dict


def main():
    parser = argparse.ArgumentParser(description="Apply Face and Object Detection to Photo Catalog")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG), help="Path to input photo catalog CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output catalog path")
    parser.add_argument("--face-model", type=str, default=str(DEFAULT_FACE_MODEL), help="Path to YuNet ONNX model")
    parser.add_argument("--yolo-model", type=str, default=str(DEFAULT_YOLO_MODEL), help="Path to YOLO model")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent processes")
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
        
    print(f"Starting inference on {total} remaining images with {args.workers} workers...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_headers = not output_path.exists()
    
    columns_to_keep = [
        "location", "folder_name", "file_name", "datetime_original",
        "orientation_label", "rotation_needed", "flip_needed",
        "face_detected", "number_of_faces", 
        "objects_detected", "objects_detected_set", "error"
    ]
    
    # Execute Multiprocessing Pool
    pool = Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.face_model, args.yolo_model, args.tile_size, args.overlap)
    )
    
    faces_detected_count = 0
    
    with open(output_path, 'a', newline='', encoding='utf-8-sig') as f:
        for i, res in enumerate(pool.imap_unordered(process_image, filtered_records), 1):
            if res.get("face_detected", False):
                faces_detected_count += 1
                
            # Filter output to only relevant join fields and detection results
            res_filtered = {col: res.get(col, "") for col in columns_to_keep}
            row_df = pd.DataFrame([res_filtered])
            
            # Ensure proper column ordering
            row_df = row_df[columns_to_keep]
            row_df.to_csv(f, header=write_headers, index=False)
            write_headers = False
            f.flush() # Ensure it writes to disk immediately
            
            if i % 10 == 0 or i == total:
                print(f"  -> Processed {i}/{total} images...")
            
    pool.close()
    pool.join()
    
    print(f"\n============================================================")
    print(f"  Detection Complete")
    print(f"============================================================")
    print(f"  Total Processed This Run:     {total}")
    print(f"  New Faces Detected This Run:  {faces_detected_count}")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
