import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from pathlib import Path
import torch
from PIL import Image
from src.data.dataset import SDCCOCODataset, get_base_data_loaders, generate_complement_training_set

class TestSDCCOCODataset(unittest.TestCase):

    @patch("src.data.dataset.Coco")
    @patch("os.path.join")
    @patch("PIL.Image.open")
    def test_sdccoco_dataset(self, mock_image_open, mock_path_join, mock_coco):
        mock_coco_instance = MagicMock()
        mock_coco.return_value = mock_coco_instance

        mock_coco_instance.imgs = {1: {"file_name": "img1.jpg"}}
        mock_coco_instance.getAnnIds.return_value = [1]
        mock_coco_instance.loadAnns.return_value = [
            {"bbox": [10, 20, 30, 40], "category_id": 1, "area": 1200, "iscrowd": 0}
        ]
        mock_coco_instance.loadImgs.return_value = [{"file_name": "img1.jpg"}]

        mock_path_join.return_value = "path/to/img1.jpg"
        mock_image_open.return_value = MagicMock()

        dataset = SDCCOCODataset(root="root_path", annotation="annotation_path")

        img, annotation, path = dataset[0]

        # Assert image and annotations are returned correctly
        self.assertEqual(path, "img1.jpg")
        self.assertEqual(len(annotation["boxes"]), 1)
        self.assertEqual(len(annotation["labels"]), 1)

    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_generate_complement_training_set(self, mock_json_load, mock_open_file, mock_json_dump):
        mock_json_load.side_effect = [
            {
                "images": [{"id": 1}, {"id": 2}],
                "annotations": [{"image_id": 1}],
                "categories": [{"id": 1, "name": "category1"}]
            },
            {
                "images": [{"id": 1}],
                "annotations": [],
                "categories": [{"id": 1, "name": "category1"}]
            }
        ]

        complement_file = generate_complement_training_set(
            data_dir="test_dir",
            training_coco="training.json",
            whole_training_coco="whole_training.json"
        )

        mock_open_file.assert_any_call("test_dir/diff_annotations.json", "w")
        self.assertEqual(complement_file, "diff_annotations.json")

    @patch("torch.utils.data.DataLoader")
    @patch("src.data.dataset.SDCCOCODataset")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_get_base_data_loaders(self, mock_json_load, mock_open_file, mock_dataset, mock_dataloader):
        mock_json_load.side_effect = [
            {
                "images": [{"id": 1}],
                "annotations": [{"image_id": 1}],
                "categories": [{"id": 1, "name": "category1"}]
            },
            {
                "images": [{"id": 1}],
                "annotations": [],
                "categories": [{"id": 1, "name": "category1"}]
            },
        ]

        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = MagicMock()

        init_loader, train_loader, complement_loader, val_loader = get_base_data_loaders(
            data_dir="test_dir", batch_size=4
        )

        self.assertTrue(mock_dataloader.called)
        self.assertEqual(mock_dataloader.call_count, 4)

if __name__ == "__main__":
    unittest.main()
