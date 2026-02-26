"""
MDETR Annotations to TSV Converter

This script converts MDETR format JSON annotation files to TSV format for benchmarking.
It supports both standard and Qwen2.5 resized coordinate modes.

MDETR annotations are expected to follow the COCO format with additional 'caption' field.
"""

import os
import csv
import base64
import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from pycocotools.coco import COCO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert MDETR format annotations to TSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python convert_to_tsv.py \\
        --images_folder /path/to/images \\
        --annotations_files /path/to/anno1.json /path/to/anno2.json \\
        --output_dir /path/to/output \\
        --qwen25

Note:
    - Annotation files must be in MDETR format (COCO format with 'caption' field)
    - Images should be accessible from the images_folder path
        """
    )
    
    parser.add_argument(
        '--images_folder',
        type=str,
        required=True,
        help='Path to the folder containing images (e.g., train2014/)'
    )
    
    parser.add_argument(
        '--annotations_files',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to MDETR format JSON annotation file(s). '
             'Multiple files can be specified separated by spaces.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the output TSV files'
    )
    
    parser.add_argument(
        '--qwen25',
        action='store_true',
        help='Use Qwen2.5 mode: resize coordinates to multiples of patch size (28). '
             'If not set, original coordinates will be used.'
    )
    
    parser.add_argument(
        '--patch_size',
        type=int,
        default=28,
        help='Patch size for resizing in Qwen2.5 mode (default: 28)'
    )
    
    return parser.parse_args()


def encode_image_to_base64(img_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If image file does not exist
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
        
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def calculate_resize_scales(height: int, width: int, patch_size: int) -> Tuple[float, float]:
    """
    Calculate resize scales for Qwen2.5 mode.
    
    In Qwen2.5 mode, images are resized to dimensions that are multiples of patch_size.
    
    Args:
        height: Original image height
        width: Original image width
        patch_size: Patch size (typically 28)
        
    Returns:
        Tuple of (scale_x, scale_y) for width and height respectively
    """
    new_h = round(height / patch_size) * patch_size
    new_w = round(width / patch_size) * patch_size
    resize_scale_x = new_w / width
    resize_scale_y = new_h / height
    return resize_scale_x, resize_scale_y


def convert_bbox_coordinates(
    bbox: List[float],
    use_qwen25: bool,
    resize_scale_x: float = 1.0,
    resize_scale_y: float = 1.0
) -> List[float]:
    """
    Convert bounding box coordinates based on the mode.
    
    Args:
        bbox: Bounding box in [x, y, w, h] format
        use_qwen25: Whether to use Qwen2.5 resized coordinates
        resize_scale_x: Scale factor for x-axis (used in Qwen2.5 mode)
        resize_scale_y: Scale factor for y-axis (used in Qwen2.5 mode)
        
    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    x1, y1, w, h = bbox
    
    if use_qwen25:
        # Qwen2.5 mode: resize coordinates to match resized image
        new_x1 = int(round(x1 * resize_scale_x))
        new_y1 = int(round(y1 * resize_scale_y))
        new_x2 = int(round((x1 + w) * resize_scale_x))
        new_y2 = int(round((y1 + h) * resize_scale_y))
        return [new_x1, new_y1, new_x2, new_y2]
    else:
        # Standard mode: use original coordinates
        return [x1, y1, x1 + w, y1 + h]


def process_annotation_file(
    annotation_file: str,
    images_folder: str,
    tsv_output_path: str,
    use_qwen25: bool = False,
    patch_size: int = 28
):
    """
    Process a single MDETR annotation file and convert to TSV format.
    
    The output TSV file has the following columns:
    - index: Image ID
    - image: Base64 encoded image
    - question: Caption/referring expression from annotations
    - height: Image height
    - width: Image width
    - answer: Space-separated bbox coordinates (x1 y1 x2 y2 ...)
    
    Args:
        annotation_file: Path to MDETR format JSON annotation file
        images_folder: Path to folder containing images
        tsv_output_path: Path to save output TSV file
        use_qwen25: Whether to use Qwen2.5 resized coordinate mode
        patch_size: Patch size for Qwen2.5 mode
    """
    print(f"Loading annotations from: {annotation_file}")
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    
    print(f"Found {len(img_ids)} images")
    print(f"Mode: {'Qwen2.5 (resized coordinates)' if use_qwen25 else 'Standard (original coordinates)'}")
    
    with open(tsv_output_path, mode='w', newline='', encoding='utf-8') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        
        # Write header
        writer.writerow(['index', 'image', 'question', 'height', 'width', 'answer'])
        
        for idx, img_id in enumerate(img_ids, 1):
            # Load image and annotation information
            raw_img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            raw_ann_info = coco.loadAnns(ann_ids)
            
            # Get image path and encode to base64
            img_path = os.path.join(images_folder, raw_img_info['file_name'])
            
            try:
                img_base64 = encode_image_to_base64(img_path)
            except FileNotFoundError as e:
                print(f"Warning: {e}, skipping...")
                continue
            
            # Get image dimensions
            height = raw_img_info['height']
            width = raw_img_info['width']
            
            # Calculate resize scales if in Qwen2.5 mode
            if use_qwen25:
                resize_scale_x, resize_scale_y = calculate_resize_scales(
                    height, width, patch_size
                )
            else:
                resize_scale_x = resize_scale_y = 1.0
            
            # Process all annotations for this image
            bbox_coords = []
            for ann in raw_ann_info:
                # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
                bbox = convert_bbox_coordinates(
                    ann['bbox'],
                    use_qwen25,
                    resize_scale_x,
                    resize_scale_y
                )
                bbox_coords.extend(bbox)  # Flatten all coordinates
            
            # Get caption (referring expression)
            # MDETR format should have 'caption' field in image info
            caption = raw_img_info.get('caption', '')
            
            # Write TSV row
            writer.writerow([
                img_id,
                img_base64,
                caption,
                height,
                width,
                ' '.join(map(str, bbox_coords))  # Join all bbox coordinates
            ])
            
            # Progress update
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(img_ids)} images...")
    
    print(f"Successfully saved to: {tsv_output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.images_folder):
        raise ValueError(f"Images folder does not exist: {args.images_folder}")
    
    for ann_file in args.annotations_files:
        if not os.path.isfile(ann_file):
            raise ValueError(f"Annotation file does not exist: {ann_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Process each annotation file
    for annotation_file in args.annotations_files:
        # Generate output filename
        base_name = Path(annotation_file).stem
        tsv_output_path = os.path.join(args.output_dir, f"{base_name}.tsv")
        
        print(f"\n{'='*60}")
        print(f"Processing: {annotation_file}")
        print(f"Output: {tsv_output_path}")
        print(f"{'='*60}")
        
        # Process the annotation file
        process_annotation_file(
            annotation_file=annotation_file,
            images_folder=args.images_folder,
            tsv_output_path=tsv_output_path,
            use_qwen25=args.qwen25,
            patch_size=args.patch_size
        )
    
    print(f"\n{'='*60}")
    print("All files processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
