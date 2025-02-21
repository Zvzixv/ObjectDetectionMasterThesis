import os

import tqdm
from pycocotools.coco import COCO

if __name__ == '__main__':
    # Load COCO annotations
    # annFile = 'path_to_annotations/instances_train2017.json'  # Replace with your path
    # output_path = 'path_to_save/filtered_annotations.json'  # Replace with your path

    train_data_dir = r'/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json/export'
    annFile = os.path.join(train_data_dir, '_annotations.coco.json')
    output_path = annFile

    coco = COCO(annFile)

    # Get all image IDs
    all_image_ids = coco.getImgIds()

    # Get image IDs that have object annotations (instances)
    annotated_image_ids = set()
    for ann in tqdm.tqdm(coco.anns.values()):
        if ann['image_id'] not in annotated_image_ids:
            annotated_image_ids.add(ann['image_id'])

    # Filter out images with no object annotations
    images_with_objects = [img_id for img_id in all_image_ids if img_id in annotated_image_ids]

    # Load filtered images information
    filtered_images_info = coco.loadImgs(images_with_objects)

    # Output the result
    print(f"Total images with objects: {len(filtered_images_info)}")

    # If you want to save the filtered images info to a new annotation file:
    import json

    filtered_annotations = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'images': filtered_images_info,
        'annotations': [ann for ann in coco.dataset['annotations'] if ann['image_id'] in images_with_objects],
        'categories': coco.dataset['categories']
    }

    # Save the filtered dataset
    with open(output_path, 'w') as outfile:
        json.dump(filtered_annotations, outfile)

    print(f"Filtered annotations saved to {output_path}")
