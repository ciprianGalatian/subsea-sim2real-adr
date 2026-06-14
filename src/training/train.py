import os
import sys
import json
import yaml
import glob
import atexit
import shutil
import subprocess
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO, RTDETR

REAL_BASELINE_MODE = False 
USE_RTDETR = True 

class TailLogger:
    def __init__(self, max_lines=200, log_file="last_200_lines.log"):
        self.terminal_out = sys.stdout
        self.terminal_err = sys.stderr
        self.log_queue = deque(maxlen=max_lines)
        self.log_file = log_file

    def write(self, message):
        self.terminal_out.write(message)
        if message.strip():
            for line in message.splitlines():
                if line.strip():
                    self.log_queue.append(line)

    def flush(self):
        self.terminal_out.flush()
        
    def save_log(self):
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"--- LAST {len(self.log_queue)} LINES OF LOG ---\n")
                for line in self.log_queue:
                    f.write(line + "\n")
            self.terminal_out.write(f"\n[Log Captured] Last {len(self.log_queue)} lines saved to: {self.log_file}\n")
        except Exception as e:
            self.terminal_err.write(f"\nFailed to save log: {e}\n")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
BASE_PATH = os.path.join(BASE_DIR, "data")
log_path = os.path.join(BASE_PATH, "run_tail_log.txt")

logger = TailLogger(max_lines=200, log_file=log_path)
sys.stdout = logger
sys.stderr = logger 
atexit.register(logger.save_log) 

def clean_yolo_zombies():
    print("\n[System Check] Sweeping for orphaned YOLO multi-GPU zombie processes...")
    try:
        subprocess.run(
            "pkill -9 -f 'Ultralytics/DDP'", 
            shell=True, 
            stderr=subprocess.DEVNULL, 
            stdout=subprocess.DEVNULL
        )
        
        ddp_dir = os.path.expanduser("~/.config/Ultralytics/DDP")
        if os.path.exists(ddp_dir):
            temp_files = glob.glob(os.path.join(ddp_dir, "_temp_*.py"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            if temp_files:
                print(f" Deleted {len(temp_files)} orphaned DDP temp files.")
    except Exception as e:
        print(f"Cleanup non-fatal error: {e}")

def select_dataset_run(base_path):
    search_pattern = os.path.join(base_path, "blenderproc_output_*")
    runs = sorted([r for r in glob.glob(search_pattern) if os.path.isdir(r)], reverse=True)
    
    if not runs:
        raise FileNotFoundError(f"No synthetic datasets found matching {search_pattern}")

    print("\n" + "="*50)
    print(" AVAILABLE SYNTHETIC DATASETS (100% TRAIN)")
    print("="*50)
    for i, run in enumerate(runs):
        print(f" [{i}] {os.path.basename(run)}")
    print("="*50)
    
    try:
        choice = input(f"\nSelect dataset index [0-{len(runs)-1}] (Press Enter for 0): ")
        choice_idx = int(choice) if choice.strip() else 0
        if choice_idx < 0 or choice_idx >= len(runs):
            print("Invalid choice, defaulting to 0.")
            choice_idx = 0
    except ValueError:
        print("Invalid input, defaulting to 0.")
        choice_idx = 0

    selected_run = runs[choice_idx]
    print(f"\nSelected Dataset: {os.path.basename(selected_run)}")
    return selected_run

def compile_chunks_to_yolo(run_dir, synth_id_map):
    yolo_base = os.path.join(run_dir, "yolo_ready_dataset")
    images_train = os.path.join(yolo_base, "images", "train")
    labels_train = os.path.join(yolo_base, "labels", "train")
    images_val = os.path.join(yolo_base, "images", "val")
    labels_val = os.path.join(yolo_base, "labels", "val")
    
    for d in [images_train, labels_train, images_val, labels_val]:
        os.makedirs(d, exist_ok=True)
    
    for cache_file in Path(yolo_base).rglob("*.cache"):
        try: os.remove(cache_file)
        except Exception: pass
            
    chunks = sorted(glob.glob(os.path.join(run_dir, "chunk_*")))
    if not chunks:
        raise FileNotFoundError(f"No chunk directories found inside {run_dir}")

    all_valid_images = []
    
    for chunk in chunks:
        chunk_name = os.path.basename(chunk) 
        coco_json_path = os.path.join(chunk, "coco_data", "coco_annotations.json")
        if not os.path.exists(coco_json_path): continue
            
        try:
            with open(coco_json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\nWARNING: Corrupt JSON detected in {chunk_name}. Skipping chunk safely.")
            continue
            
        for img_info in data.get('images', []):
            img_id = img_info['id']
            anns = [a for a in data.get('annotations', []) if a['image_id'] == img_id]
            
            valid_anns = [a for a in anns if a['category_id'] in synth_id_map]
            if not valid_anns: continue 
                
            all_valid_images.append({
                'chunk_name': chunk_name,
                'chunk_dir': chunk,
                'img_info': img_info,
                'anns': valid_anns
            })

    import random
    random.shuffle(all_valid_images)
    split_index = int(len(all_valid_images) * 0.8)
    
    train_images = all_valid_images[:split_index]
    val_images = all_valid_images[split_index:]

    print(f"\nCompiling Synthetic Dataset: 80/20 Split ({len(train_images)} Train | {len(val_images)} Val)...")

    def process_split(image_list, out_images_dir, out_labels_dir):
        boxes_written = 0
        for item in image_list:
            img_info = item['img_info']
            chunk_name = item['chunk_name']
            chunk_dir = item['chunk_dir']
            anns = item['anns']
            
            img_w, img_h = img_info['width'], img_info['height']
            rel_img_path = img_info['file_name']
            src_img_path = os.path.join(chunk_dir, "coco_data", rel_img_path)
            orig_file_name = os.path.basename(rel_img_path)
            
            new_file_name = f"{chunk_name}_{orig_file_name}"
            txt_name = new_file_name.replace('.jpg', '.txt').replace('.png', '.txt')
            
            dest_txt_path = os.path.join(out_labels_dir, txt_name)
            dest_img_path = os.path.join(out_images_dir, new_file_name)

            if os.path.exists(src_img_path):
                with open(dest_txt_path, 'w') as out_f:
                    for ann in anns:
                        yolo_id = synth_id_map[ann['category_id']]
                        x_min, y_min, bw, bh = ann['bbox']
                        
                        if bw > 1.0 or bh > 1.0:
                            x_center = (x_min + bw / 2.0) / img_w
                            y_center = (y_min + bh / 2.0) / img_h
                            norm_w, norm_h = bw / img_w, bh / img_h
                        else:
                            x_center, y_center = x_min + (bw / 2.0), y_min + (bh / 2.0)
                            norm_w, norm_h = bw, bh
                        
                        out_f.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        boxes_written += 1
                
                if not os.path.exists(dest_img_path):
                    os.symlink(src_img_path, dest_img_path)
        return boxes_written

    train_boxes = process_split(train_images, images_train, labels_train)
    val_boxes = process_split(val_images, images_val, labels_val)
    
    print(f"Written {train_boxes} train boxes and {val_boxes} val boxes.")
    return yolo_base, images_train, images_val

def align_test_dataset(real_base_dir, yolo_base_dir):
    print("\n--- COMPILING TEST SET (COMBINING TRAIN/VAL) ---")
    
    test_images_out = os.path.join(yolo_base_dir, "images", "test")
    test_labels_out = os.path.join(yolo_base_dir, "labels", "test")
    os.makedirs(test_images_out, exist_ok=True)
    os.makedirs(test_labels_out, exist_ok=True)
    
    processed_pure = 0
    
    for split in ["train", "val"]:
        real_labels_dir = os.path.join(real_base_dir, "labels", split)
        real_images_dir = os.path.join(real_base_dir, "images", split)
        
        if not os.path.exists(real_labels_dir):
            continue
        
        for label_file in glob.glob(os.path.join(real_labels_dir, "*.txt")):
            base_name = os.path.basename(label_file)
            
            src_label = label_file
            dst_label = os.path.join(test_labels_out, base_name)
            shutil.copy2(src_label, dst_label)
                
            for ext in ['.jpg', '.jpeg', '.png']:
                img_name = base_name.replace('.txt', ext)
                src_img = os.path.join(real_images_dir, img_name)
                if os.path.exists(src_img):
                    dst_img = os.path.join(test_images_out, img_name)
                    if not os.path.exists(dst_img):
                        os.symlink(src_img, dst_img)
                    break
                    
            processed_pure += 1
            
    print(f"Successfully combined {processed_pure} REAL images into the TEST set.")
    return test_images_out

def create_yaml_config(yaml_path, train_dir, val_dir, test_dir, class_names):
    dataset_config = {
        "train": train_dir, 
        "val": val_dir,   
        "test": test_dir,          
        "nc": len(class_names),                    
        "names": class_names          
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, sort_keys=False)

def main():
    clean_yolo_zombies()

    PROJECT_DIR = os.path.join(BASE_PATH, "training_runs")
    
    pure_real_base_dir = os.path.join(BASE_PATH, "pure_yolo_dataset")
    golden_classes_txt = os.path.join(pure_real_base_dir, "classes.txt")
    
    if not os.path.exists(golden_classes_txt):
        print(f"\nCRITICAL ERROR: Pure dataset missing at {golden_classes_txt}")
        sys.exit(1)
        
    with open(golden_classes_txt, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
        
    golden_name_to_id = {name.lower().replace(" ", "_").replace("-", "_"): i for i, name in enumerate(class_names)}
    
    arch_prefix = "RTDETR" if USE_RTDETR else "YOLOv8s"

    if REAL_BASELINE_MODE:
        print("\n" + "="*50)
        print(f" REAL BASELINE MODE ENABLED ({arch_prefix})")
        print(" Bypassing synthetic menus. Model will train and validate ")
        print(" STRICTLY on the physical 'pure_yolo_dataset' images.")
        print("="*50)
        
        synthetic_train_images_dir = os.path.join(pure_real_base_dir, "images", "train")
        synthetic_val_images_dir = os.path.join(pure_real_base_dir, "images", "val")
        real_test_images_dir = os.path.join(pure_real_base_dir, "images", "val") 
        
        unique_run_name = f"REAL_DATA_BASELINE_{arch_prefix}"
        sim2real_yaml = os.path.join(pure_real_base_dir, "real_baseline.yaml")
        print(f"Detected {len(class_names)} Golden classes: {class_names}")
        
    else:
        selected_run_dir = select_dataset_run(BASE_PATH)
        
        synth_classes_txt_path = os.path.join(selected_run_dir, "classes.txt")
        synth_id_map = {} 
        
        if os.path.exists(synth_classes_txt_path):
            with open(synth_classes_txt_path, "r") as f:
                for line in f:
                    if ":" in line:
                        bp_id_str, c_name = line.split(":", 1)
                        bp_id = int(bp_id_str.strip())
                        norm_c_name = c_name.strip().lower().replace(" ", "_").replace("-", "_")
                        if norm_c_name in golden_name_to_id:
                            synth_id_map[bp_id] = golden_name_to_id[norm_c_name]
        else:
            print("\nWARNING: classes.txt not found in synthetic dataset.")
            sys.exit(1)

        print(f"Detected {len(class_names)} Golden classes: {class_names}")
        
        yolo_base_dir, synthetic_train_images_dir, synthetic_val_images_dir = compile_chunks_to_yolo(selected_run_dir, synth_id_map)
        
        real_test_images_dir = align_test_dataset(
            real_base_dir=pure_real_base_dir, 
            yolo_base_dir=yolo_base_dir
        )
        
        valid_exts = {'.jpg', '.jpeg', '.png'}
        real_images_found = [f for f in os.listdir(real_test_images_dir) if os.path.splitext(f)[1].lower() in valid_exts] if os.path.exists(real_test_images_dir) else []
        
        if not real_images_found:
            print("\n" + "!"*60)
            print(" CRITICAL ERROR: NO REAL IMAGES FOUND FOR TEST")
            sys.exit(1)

        print(f"\nUsing {len(real_images_found)} REAL images for Final Test.")
        
        unique_run_name = f"multi_class_sim2real_{arch_prefix}_{os.path.basename(selected_run_dir)}"
        sim2real_yaml = os.path.join(selected_run_dir, "pure_sim2real.yaml")

    create_yaml_config(sim2real_yaml, train_dir=synthetic_train_images_dir, val_dir=synthetic_val_images_dir, test_dir=real_test_images_dir, class_names=class_names)
    
    print(f"\n--- STARTING {arch_prefix} MULTI-CLASS SIM-TO-REAL TRAINING ---")
    
    if USE_RTDETR:
        print("[Architecture] Initializing RT-DETR-L (Transformer Vision Model)...")
        model_stage2 = RTDETR("rtdetr-l.pt")
    else:
        print("[Architecture] Initializing YOLOv8s (CNN Vision Model)...")
        model_stage2 = YOLO("yolov8s.pt") 
    
    train_args = {
        "data": sim2real_yaml,
        "epochs": 300,            
        "imgsz": 640,              
        "batch": 16,            
        "device": [0, 1],          
        "project": PROJECT_DIR, 
        "name": unique_run_name,  
        "plots": True,
        "optimizer": 'AdamW',      
        "cos_lr": True,            
        "lrf": 0.01,                
        "warmup_epochs": 5,        
        "patience": 30             
    }

    if USE_RTDETR:
        print("injecting optimal Transformer hyperparams (Lower LR, standard decay).")
        train_args["lr0"] = 0.0001
        train_args["weight_decay"] = 0.0001
    else:
        print("Injecting optimal CNN hyperparams (Higher LR, heavy decay, frozen backbone).")
        train_args["lr0"] = 0.001
        train_args["weight_decay"] = 0.01
        train_args["freeze"] = 10
        train_args["label_smoothing"] = 0.1

    results = model_stage2.train(**train_args)

    def harvest_ablation_metrics(model, dataset_yaml, run_id, output_dir, project_dir):
        print(f"\n Running strict COCO evaluation on TEST split for {run_id}...")
        
        val_metrics = model.val(
            data=dataset_yaml, 
            split='test', 
            save_json=True,
            project=os.path.join(project_dir, run_id),
            name="test_evaluation"
        )
        
        map50 = val_metrics.box.map50
        map50_95 = val_metrics.box.map
        
        try:
            mean_recall = val_metrics.box.mr 
        except:
            mean_recall = 0.0

        cm = val_metrics.confusion_matrix.matrix
        background_row = cm[-1, :-1]
        total_false_positives = np.sum(background_row)
        total_predictions = np.sum(cm[:, :-1])
        
        fpr = total_false_positives / total_predictions if total_predictions > 0 else 0.0

        metrics_ledger = {
            "Run_ID": run_id,
            "mAP_50": float(map50),
            "mAP_50_95": float(map50_95),
            "Recall": float(mean_recall), 
            "FPR": float(fpr),
            "Total_False_Positives": int(total_false_positives)
        }

        save_path = os.path.join(output_dir, f"{run_id}_metrics.json")
        with open(save_path, "w") as f:
            json.dump(metrics_ledger, f, indent=4)
            
        local_save_path = os.path.join(project_dir, run_id, "test_metrics_summary.json")
        with open(local_save_path, "w") as f:
            json.dump(metrics_ledger, f, indent=4)
            
        print(f"[✓] Metrics safely harvested and saved to {save_path} and {local_save_path}")

    best_weights_path = os.path.join(PROJECT_DIR, unique_run_name, "weights", "best.pt")
    
    print(f"\nLoading BEST weights from {best_weights_path} to ensure pristine metrics...")
    
    best_model = RTDETR(best_weights_path) if USE_RTDETR else YOLO(best_weights_path)

    metrics_output_dir = os.path.join(BASE_PATH, "metrics_output")
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    harvest_ablation_metrics(best_model, sim2real_yaml, unique_run_name, metrics_output_dir, PROJECT_DIR)

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Final weights: {best_weights_path}")

if __name__ == "__main__":
    main()