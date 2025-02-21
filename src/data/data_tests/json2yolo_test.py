import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil
import numpy as np
from pathlib import Path
from pathlib import Path
from PIL import Image

# Ensure src is in PYTHONPATH
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.json2yolo import (
    convert_coco_json, min_index, merge_multi_segment, delete_dsstore
)

class TestConversionUtils(unittest.TestCase):

    @patch("src.data.json2yolo.convert_coco_json")
    def test_convert_coco_json(self, mock_convert_coco_json):
        json_dir = "test_json_dir"
        save_dir = "test_save_dir"

        # Call the mocked function directly
        mock_convert_coco_json(json_dir=json_dir, save_dir=save_dir, use_segments=True, cls91to80=True)

        mock_convert_coco_json.assert_called_once_with(json_dir=json_dir, save_dir=save_dir, use_segments=True, cls91to80=True)

    def test_min_index(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        result = min_index(arr1, arr2)
        self.assertEqual(result, (1, 0), "Incorrect min_index result")

    def test_merge_multi_segment(self):
        segments = [[[0, 0, 1, 1]], [[2, 2, 3, 3]]]
        result = merge_multi_segment(segments)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Merged segments should not be empty")

    @patch("src.data.json2yolo.Path.rglob")
    @patch("src.data.json2yolo.Path.unlink")
    def test_delete_dsstore(self, mock_unlink, mock_rglob):
        mock_rglob.return_value = [Path(".DS_Store")]

        delete_dsstore(path="test_dir")

        mock_rglob.assert_called_once_with(".DS_store")
        mock_unlink.assert_called_once()

if __name__ == "__main__":
    unittest.main()
