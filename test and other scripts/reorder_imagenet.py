import os
import xml.etree.ElementTree as ET
from shutil import move
from tqdm import tqdm

# === Paths ===
images_dir = r"D:\imagenet\ILSVRC\Data\CLS-LOC\val"  # all images
annotations_dir = r"D:\imagenet\ILSVRC\Annotations\CLS-LOC\val"  # XMLs
output_dir = images_dir  # we create class subfolders here

# Build a set of all images in the folder (lowercase for matching)
all_images = {f.lower(): f for f in os.listdir(images_dir) if f.lower().endswith(".jpeg")}

# Step 1: Parse annotations and map filename -> class
file_to_class = {}
for ann_file in tqdm(os.listdir(annotations_dir), desc="Parsing XML annotations"):
    if not ann_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(annotations_dir, ann_file))
    root = tree.getroot()
    filename = root.find("filename").text
    if not filename.lower().endswith(".jpeg"):
        filename_ext = filename + ".JPEG"
    else:
        filename_ext = filename
    file_to_class[filename_ext] = root.find("object").find("name").text

# Step 2: Move images into class folders
missing_files = []
for filename, class_id in tqdm(file_to_class.items(), desc="Moving images"):
    src_name = all_images.get(filename.lower())
    if not src_name:
        missing_files.append(filename)
        continue

    src = os.path.join(images_dir, src_name)
    class_folder = os.path.join(output_dir, class_id)
    os.makedirs(class_folder, exist_ok=True)
    dst = os.path.join(class_folder, src_name)
    move(src, dst)

if missing_files:
    print(f"Warning: {len(missing_files)} files not found, e.g., {missing_files[:5]}")
