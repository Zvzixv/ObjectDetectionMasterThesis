import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.consts import N_TRAIN
from src.data.yolo_utils import clean_yolo_dataset, check_for_data_leaks

class TestYoloUtils(unittest.TestCase):

    @patch("shutil.rmtree")
    @patch("shutil.copytree")
    @patch("os.listdir")
    def test_clean_yolo_dataset(self, mock_listdir, mock_copytree, mock_rmtree):
        mock_listdir.side_effect = [
            ["img1.jpg", "img2.jpg"],  # AL_TRAIN_DIR
            ["img3.jpg", "img4.jpg"],  # AL_UNLABELED_DIR
            ["img5.jpg", "img6.jpg"],  # INIT/images
            ["img7.jpg", "img8.jpg"],  # DIFF/images
        ]
        mock_logger = MagicMock()

        clean_yolo_dataset("test_yolo_path", mock_logger)

        # Assert directories were cleaned
        mock_rmtree.assert_any_call(Path("test_yolo_path") / "active_learning_train" / "images")
        mock_rmtree.assert_any_call(Path("test_yolo_path") / "active_learning_unlabeled" / "images")

        # Assert copytree was called for INIT and DIFF directories
        mock_copytree.assert_any_call(
            Path("test_yolo_path") / "init" / "images",
            Path("test_yolo_path") / "active_learning_train" / "images",
        )
        mock_copytree.assert_any_call(
            Path("test_yolo_path") / "diff" / "images",
            Path("test_yolo_path") / "active_learning_unlabeled" / "images",
        )

        # Assert N_TRAIN consistency
        mock_logger.info.assert_any_call("Cleaning the AL datasets.")
        self.assertEqual(mock_logger.info.call_count, 3)

    @patch("os.listdir")
    def test_check_for_data_leaks(self, mock_listdir):
        mock_listdir.side_effect = [
            ["img1.jpg", "img2.jpg"],  # train/images
            ["img3.jpg", "img4.jpg"],  # val/images
            ["img5.jpg", "img6.jpg"],  # test/images
            ["img7.jpg", "img8.jpg"],  # init/images
            ["img9.jpg", "img10.jpg"],  # diff/images
            ["img11.jpg", "img12.jpg"],  # active_learning_train/images
            ["img13.jpg", "img14.jpg"],  # active_learning_unlabeled/images
            ["img15.jpg", "img16.jpg"],  # Extra for train/val check
            ["img17.jpg", "img18.jpg"],  # Extra for val/test check
            ["img19.jpg", "img20.jpg"],  # Extra for train/test check
            ["img21.jpg", "img22.jpg"],  # Extra for init/diff check
            ["img23.jpg", "img24.jpg"],  # Extra for active_learning_train/unlabeled check
        ]
        mock_logger = MagicMock()

        check_for_data_leaks("test_yolo_path", mock_logger)

        # Assert logger calls for no data leaks
        self.assertEqual(mock_logger.info.call_count, 5)
        mock_logger.error.assert_not_called()

if __name__ == "__main__":
    unittest.main()
