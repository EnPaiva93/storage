import json
from PIL import Image
import io
from tqdm import tqdm
import os
import json
import shutil
from datasets import load_dataset

def convert_to_coco(dataset):
    """
    Converts a HuggingFace-style dataset into a COCO JSON structure
    and simplifies the category mapping into three classes:
      0 -> text, 1 -> figure, 2 -> table.
    
    Each dataset item is expected to be a dictionary with:
      - "filename": image filename,
      - "image_data": raw image bytes,
      - "anno_string": a list of annotation strings where each string's first token is
                       the category id and subsequent tokens include normalized bbox coordinates.
    
    Returns a simplified COCO dictionary.
    """
    # --- Step 1: Detailed Conversion ---
    detailed_coco = {
        "info": {
            "description": "Converted dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "categories": [
            {'id': 0, 'name': 'QR code', 'supercategory': ''},
            {'id': 1, 'name': 'advertisement', 'supercategory': ''},
            {'id': 2, 'name': 'algorithm', 'supercategory': ''},
            {'id': 3, 'name': 'answer', 'supercategory': ''},
            {'id': 4, 'name': 'author', 'supercategory': ''},
            {'id': 5, 'name': 'barcode', 'supercategory': ''},
            {'id': 6, 'name': 'bill', 'supercategory': ''},
            {'id': 7, 'name': 'blank', 'supercategory': ''},
            {'id': 8, 'name': 'bracket', 'supercategory': ''},
            {'id': 9, 'name': 'breakout', 'supercategory': ''},
            {'id': 10, 'name': 'byline', 'supercategory': ''},
            {'id': 11, 'name': 'caption', 'supercategory': ''},
            {'id': 12, 'name': 'catalogue', 'supercategory': ''},
            {'id': 13, 'name': 'chapter title', 'supercategory': ''},
            {'id': 14, 'name': 'code', 'supercategory': ''},
            {'id': 15, 'name': 'correction', 'supercategory': ''},
            {'id': 16, 'name': 'credit', 'supercategory': ''},
            {'id': 17, 'name': 'dateline', 'supercategory': ''},
            {'id': 18, 'name': 'drop cap', 'supercategory': ''},
            {'id': 19, 'name': "editor's note", 'supercategory': ''},
            {'id': 20, 'name': 'endnote', 'supercategory': ''},
            {'id': 21, 'name': 'examinee information', 'supercategory': ''},
            {'id': 22, 'name': 'fifth-level title', 'supercategory': ''},
            {'id': 23, 'name': 'figure', 'supercategory': ''},
            {'id': 24, 'name': 'first-level question number', 'supercategory': ''},
            {'id': 25, 'name': 'first-level title', 'supercategory': ''},
            {'id': 26, 'name': 'flag', 'supercategory': ''},
            {'id': 27, 'name': 'folio', 'supercategory': ''},
            {'id': 28, 'name': 'footer', 'supercategory': ''},
            {'id': 29, 'name': 'footnote', 'supercategory': ''},
            {'id': 30, 'name': 'formula', 'supercategory': ''},
            {'id': 31, 'name': 'fourth-level section title', 'supercategory': ''},
            {'id': 32, 'name': 'fourth-level title', 'supercategory': ''},
            {'id': 33, 'name': 'header', 'supercategory': ''},
            {'id': 34, 'name': 'headline', 'supercategory': ''},
            {'id': 35, 'name': 'index', 'supercategory': ''},
            {'id': 36, 'name': 'inside', 'supercategory': ''},
            {'id': 37, 'name': 'institute', 'supercategory': ''},
            {'id': 38, 'name': 'jump line', 'supercategory': ''},
            {'id': 39, 'name': 'kicker', 'supercategory': ''},
            {'id': 40, 'name': 'lead', 'supercategory': ''},
            {'id': 41, 'name': 'marginal note', 'supercategory': ''},
            {'id': 42, 'name': 'matching', 'supercategory': ''},
            {'id': 43, 'name': 'mugshot', 'supercategory': ''},
            {'id': 44, 'name': 'option', 'supercategory': ''},
            {'id': 45, 'name': 'ordered list', 'supercategory': ''},
            {'id': 46, 'name': 'other question number', 'supercategory': ''},
            {'id': 47, 'name': 'page number', 'supercategory': ''},
            {'id': 48, 'name': 'paragraph', 'supercategory': ''},
            {'id': 49, 'name': 'part', 'supercategory': ''},
            {'id': 50, 'name': 'play', 'supercategory': ''},
            {'id': 51, 'name': 'poem', 'supercategory': ''},
            {'id': 52, 'name': 'reference', 'supercategory': ''},
            {'id': 53, 'name': 'sealing line', 'supercategory': ''},
            {'id': 54, 'name': 'second-level question number', 'supercategory': ''},
            {'id': 55, 'name': 'second-level title', 'supercategory': ''},
            {'id': 56, 'name': 'section', 'supercategory': ''},
            {'id': 57, 'name': 'section title', 'supercategory': ''},
            {'id': 58, 'name': 'sidebar', 'supercategory': ''},
            {'id': 59, 'name': 'sub section title', 'supercategory': ''},
            {'id': 60, 'name': 'subhead', 'supercategory': ''},
            {'id': 61, 'name': 'subsub section title', 'supercategory': ''},
            {'id': 62, 'name': 'supplementary note', 'supercategory': ''},
            {'id': 63, 'name': 'table', 'supercategory': ''},
            {'id': 64, 'name': 'table caption', 'supercategory': ''},
            {'id': 65, 'name': 'table note', 'supercategory': ''},
            {'id': 66, 'name': 'teasers', 'supercategory': ''},
            {'id': 67, 'name': 'third-level question number', 'supercategory': ''},
            {'id': 68, 'name': 'third-level title', 'supercategory': ''},
            {'id': 69, 'name': 'title', 'supercategory': ''},
            {'id': 70, 'name': 'translator', 'supercategory': ''},
            {'id': 71, 'name': 'underscore', 'supercategory': ''},
            {'id': 72, 'name': 'unordered list', 'supercategory': ''},
            {'id': 73, 'name': 'weather forecast', 'supercategory': ''}
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 1
    annotation_id = 1
    for item in tqdm(dataset, desc="Converting to COCO format"):
        filename = item["filename"]
        image_data = item["image_data"]
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        # Add image entry
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": ""
        }
        detailed_coco["images"].append(image_info)
        
        # Process each annotation string for the image
        for anno_str in item["anno_string"]:
            parts = anno_str.split()
            if len(parts) < 9:
                continue  # Skip invalid entries
            # The first token is the original category id (as per detailed categories)
            category_id = int(parts[0])
            x0 = float(parts[1])
            y0 = float(parts[2])
            x1 = float(parts[5])
            y1 = float(parts[6])
            
            # Convert normalized coordinates to absolute coordinates
            abs_x0 = x0 * width
            abs_y0 = y0 * height
            abs_x1 = x1 * width
            abs_y1 = y1 * height
            bbox_width = abs_x1 - abs_x0
            bbox_height = abs_y1 - abs_y0
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,  # to be remapped below
                "bbox": [abs_x0, abs_y0, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "segmentation": [],
                "iscrowd": 0
            }
            detailed_coco["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1

    # --- Step 2: Simplify Category Mapping ---
    # Define keywords for remapping
    image_keywords = ['image', 'figure', 'mugshot', 'advertisement', 'qr code', 'barcode', 'blank', 'weather forecast', 'flag']
    table_keywords = ['table', 'catalogue']

    # Create a mapping from original category id to simplified category id
    category_id_remap = {}
    for cat in detailed_coco["categories"]:
        name = cat["name"].lower()
        cat_id = cat["id"]
        if any(keyword in name for keyword in table_keywords):
            category_id_remap[cat_id] = 2  # table
        elif any(keyword in name for keyword in image_keywords):
            category_id_remap[cat_id] = 1  # figure
        else:
            category_id_remap[cat_id] = 0  # text

    # Update annotations to use the simplified category ids
    for anno in detailed_coco["annotations"]:
        old_cat_id = anno["category_id"]
        anno["category_id"] = category_id_remap.get(old_cat_id, 0)
    
    # Build the final simplified COCO structure with new categories
    simplified_coco = {
        "info": detailed_coco.get("info", {}),
        "licenses": detailed_coco.get("licenses", []),
        "categories": [
            {"id": 0, "name": "text", "supercategory": ""},
            {"id": 1, "name": "figure", "supercategory": ""},
            {"id": 2, "name": "table", "supercategory": ""}
        ],
        "images": detailed_coco["images"],
        "annotations": detailed_coco["annotations"]
    }
    
    return simplified_coco

def prepare_coco_directory_structure(dataset, output_dir="laydoc", test_size=0.1):
    # Crear directorios
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val_images"), exist_ok=True)

    # Dividir dataset en train y val
    train_val_data = dataset.train_test_split(test_size=test_size)
    train_data = train_val_data["train"]
    val_data = train_val_data["test"]

    # Guardar imágenes de entrenamiento
    for item in tqdm(train_data, desc="Guardando Imagenes de Entrenamiento"):
        img_path = os.path.join(output_dir, "train_images", item["filename"])
        Image.open(io.BytesIO(item['image_data'])).save(img_path)

    # Procesar conjunto de entrenamiento
    train_coco = convert_to_coco(train_data)
    with open(os.path.join(output_dir, "annotations", "train.json"), 'w') as f:
        json.dump(train_coco, f)

    # Procesar conjunto de validación
    val_coco = convert_to_coco(val_data)
    with open(os.path.join(output_dir, "annotations", "val.json"), 'w') as f:
        json.dump(val_coco, f)

    # Guardar imágenes de validación
    for item in tqdm(val_data, desc="Guardando Imagenes de Validación"):
        img_path = os.path.join(output_dir, "val_images", item["filename"])
        Image.open(io.BytesIO(item['image_data'])).save(img_path)

    print(f"Dataset preparado en: {output_dir}")
    print(f"- Imágenes de entrenamiento: {len(train_data)}")
    print(f"- Imágenes de validación: {len(val_data)}")


if __name__ == "__main__":
    dataset = load_dataset("parquet", data_files={'train': [f'./docsynth60k/part{i}.parquet' for i in range(6)]})
    print(dataset)
    prepare_coco_directory_structure(dataset["train"])