import os

import torch
import torch.utils.data
import torchvision
import tqdm as tqdm
from PIL import Image
from jsonargparse import CLI
from pycocotools.coco import COCO

from src.experiment_configs import ExperimentConfig
from src.utils import collate_fn


# Common Object Classification
class SDCCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # TODO remove
        assert len(ann_ids) > 0, "No object"
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation, path

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = [
        torchvision.transforms.ToTensor()
    ]
    return torchvision.transforms.Compose(custom_transforms)


def generate_complement_training_set(data_dir: str, training_coco: str,
                                     whole_training_coco: str):
    import json

    def get_image_ids(coco_data):
        return {img['id'] for img in coco_data['images']}

    def compute_set_difference(coco_data1, coco_data2):
        image_ids1 = get_image_ids(coco_data1)
        image_ids2 = get_image_ids(coco_data2)

        diff_image_ids = image_ids1 - image_ids2

        diff_images = [img for img in coco_data1['images'] if
                       img['id'] in diff_image_ids]
        diff_annotations = [ann for ann in coco_data1['annotations'] if
                            ann['image_id'] in diff_image_ids]

        diff_data = {
            'images': diff_images,
            'annotations': diff_annotations,
            'categories': coco_data1['categories']
            # Assuming the categories are the same
        }

        return diff_data

    # Load the COCO annotations files
    with open(whole_training_coco, "r") as f:
        coco_data1 = json.load(f)

    with open(training_coco, "r") as f:
        coco_data2 = json.load(f)

    # Compute the set difference
    diff_data = compute_set_difference(coco_data1, coco_data2)

    # Save the result to a new file
    complement_name = "diff_annotations.json"
    diff_annotations = os.path.join(data_dir, complement_name)
    with open(diff_annotations, "w") as f:
        json.dump(diff_data, f)

    return complement_name


def get_base_data_loaders(data_dir: str, batch_size: int):
    init_coco = os.path.join(data_dir, 'init_annotations.json')
    train_coco = os.path.join(data_dir, 'train_annotations.json')
    whole_training_coco = os.path.join(data_dir, 'train_annotations.json')
    val_coco = os.path.join(data_dir, 'val_annotations.json')
    # create own Dataset
    init_dataset = SDCCOCODataset(root=data_dir,
                                  annotation=init_coco,
                                  transforms=get_transform()
                                  )
    train_dataset = SDCCOCODataset(root=data_dir,
                                   annotation=train_coco,
                                   transforms=get_transform()
                                   )
    val_dataset = SDCCOCODataset(root=data_dir,
                                 annotation=val_coco,
                                 transforms=get_transform()
                                 )
    complement_name = generate_complement_training_set(data_dir=data_dir,
                                                       training_coco=init_coco,
                                                       whole_training_coco=whole_training_coco)

    complement_coco = os.path.join(data_dir, complement_name)

    complement_dataset = SDCCOCODataset(root=data_dir,
                                        annotation=complement_coco,
                                        transforms=get_transform()
                                        )

    # own DataLoader
    init_data_loader = torch.utils.data.DataLoader(init_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=collate_fn)
    # own DataLoader
    complement_data_loader = torch.utils.data.DataLoader(complement_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         collate_fn=collate_fn)
    return init_data_loader, train_data_loader, complement_data_loader, val_data_loader


def get_test_data_loaders(data_dir, batch_size):
    val_coco = os.path.join(data_dir, 'test_annotations.json')
    val_dataset = SDCCOCODataset(root=data_dir,
                                 annotation=val_coco,
                                 transforms=get_transform()
                                 )
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)
    return val_data_loader


def test_dataset(experiment_config):
    init_data_loader, train_data_loader, complement_data_loader, val_data_loader = get_base_data_loaders(
        data_dir=experiment_config.data_config.rcnn_data_path,
        batch_size=experiment_config.batch_size)
    # select device (whether GPU or CPU)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DataLoader is iterable over Dataset
    for imgs, annotations, _ in tqdm.tqdm(train_data_loader):
        imgs = list(img.to(device) for img in imgs)
        print(annotations)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in
                       annotations]


if __name__ == '__main__':
    experiment_config = CLI(ExperimentConfig)
    test_dataset(experiment_config)
