import json
import os
import random


# Każdy model trenuję 8 razy używając kolejno 10% - 15 % - 20 - 25 - 30 - 40 - 60 - 80% danych o treningu
def split_coco_annotations(coco_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, init_ratio=0.1, seed=None):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Optionally, set a seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Shuffle the images
    images = coco_data['images']
    random.shuffle(images)

    # Split images into train, val, test
    total_images = len(images)
    init_end = int(init_ratio * total_images)
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    init_images = images[:init_end]
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create a helper function to filter annotations
    def filter_annotations(images_split):
        image_ids = {img['id'] for img in images_split}
        annotations_split = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        return annotations_split

    # Filter annotations for each split
    init_annotations = filter_annotations(init_images)
    train_annotations = filter_annotations(train_images)
    val_annotations = filter_annotations(val_images)
    test_annotations = filter_annotations(test_images)

    # Create the COCO formatted dictionaries for each split
    init_data = {
        'images': init_images,
        'annotations': init_annotations,
        'categories': coco_data['categories']
    }
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }
    test_data = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    return init_data, train_data, val_data, test_data


if __name__ == '__main__':
    # Load the COCO annotations file
    train_data_dir = r'/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json/export'
    train_coco = os.path.join(train_data_dir, '_annotations.coco.json')
    with open(train_coco, "r") as f:
        coco_data = json.load(f)

    # Split the data
    init_data, train_data, val_data, test_data = split_coco_annotations(coco_data)

    # Save the datasets to separate JSON files
    with open(os.path.join(train_data_dir, "init_annotations.json"), "w") as f:
        json.dump(init_data, f)

    with open(os.path.join(train_data_dir, "train_annotations.json"), "w") as f:
        json.dump(train_data, f)

    with open(os.path.join(train_data_dir, "val_annotations.json"), "w") as f:
        json.dump(val_data, f)

    with open(os.path.join(train_data_dir, "test_annotations.json"), "w") as f:
        json.dump(test_data, f)
