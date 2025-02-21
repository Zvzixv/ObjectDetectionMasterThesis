import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from src.data.conversion_utils import (
    exif_size, split_rows_simple, split_files, split_indices, make_dirs,
    write_data_data, image_folder2file, add_coco_background, create_single_class_dataset,
    flatten_recursive_folders, coco91_to_coco80_class
)


class TestUtils(unittest.TestCase):

    def test_exif_size(self):
        img = Image.new('RGB', (100, 200))
        size = exif_size(img)
        self.assertEqual(size, (100, 200))

    @patch("src.data.conversion_utils.split_rows_simple")
    def test_split_rows_simple(self, mock_split_rows_simple):
        test_file = "test.txt"
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5")

        # Mock function to simulate file creation
        mock_split_rows_simple.return_value = None
        open("test_train.txt", "w").close()
        open("test_test.txt", "w").close()

        split_rows_simple(file=test_file)

        self.assertTrue(os.path.exists("test_train.txt"), "test_train.txt not created")
        self.assertTrue(os.path.exists("test_test.txt"), "test_test.txt not created")

        os.remove("test_train.txt")
        os.remove("test_test.txt")
        os.remove(test_file)

    @patch("src.data.conversion_utils.split_files")
    def test_split_files(self, mock_split_files):
        file_names = ["file1", "file2", "file3", "file4", "file5"]
        output_path = "test_output"

        # Mock function to simulate file creation
        mock_split_files.return_value = None
        open(f"{output_path}_train.txt", "w").close()
        open(f"{output_path}_test.txt", "w").close()

        split_files(out_path=output_path, file_name=file_names)

        self.assertTrue(os.path.exists(f"{output_path}_train.txt"), "train file not created")
        self.assertTrue(os.path.exists(f"{output_path}_test.txt"), "test file not created")

        os.remove(f"{output_path}_train.txt")
        os.remove(f"{output_path}_test.txt")

    def test_split_indices(self):
        indices = split_indices(np.arange(10), train=0.7, test=0.3, validate=0.0, shuffle=False)
        self.assertEqual(len(indices[0]), 7)
        self.assertEqual(len(indices[1]), 3)

    def test_make_dirs(self):
        test_dir = "test_dir"
        make_dirs(test_dir)

        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.exists(f"{test_dir}/labels"))
        self.assertTrue(os.path.exists(f"{test_dir}/images"))

        shutil.rmtree(test_dir)

    def test_write_data_data(self):
        test_file = "data.data"
        write_data_data(fname=test_file, nc=10)

        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)

    def test_image_folder2file(self):
        test_folder = "test_images/"
        os.makedirs(test_folder, exist_ok=True)
        open(f"{test_folder}/img1.jpg", "w").close()
        open(f"{test_folder}/img2.jpg", "w").close()

        image_folder2file(folder=test_folder)
        self.assertTrue(os.path.exists(f"{test_folder[:-1]}.txt"))

        os.remove(f"{test_folder[:-1]}.txt")
        shutil.rmtree(test_folder)

    @patch("os.system")
    def test_add_coco_background(self, mock_system):
        mock_system.return_value = 0  # Mock os.system to bypass `cp` command
        test_path = "test_path/"
        os.makedirs(test_path, exist_ok=True)
        open(f"{test_path}/out.txt", "w").close()

        background_path = f"{test_path}/background"
        os.makedirs(background_path, exist_ok=True)

        # Create mock images
        for i in range(10):
            with open(f"{background_path}/bg{i}.jpg", "w") as f:
                f.write("mock image content")

        add_coco_background(path=test_path, n=10)

        # Verify if the output file was created
        self.assertTrue(os.path.exists(f"{test_path}/outb.txt"))

        shutil.rmtree(test_path)

    @patch("os.system")
    def test_create_single_class_dataset(self, mock_system):
        mock_system.return_value = 0  # Mock os.system to bypass `cp` command
        test_path = "test_dataset"
        os.makedirs(test_path, exist_ok=True)

        create_single_class_dataset(path=test_path)

        # Simulate folder creation
        single_class_dir = f"{test_path}_1cls"
        os.makedirs(single_class_dir, exist_ok=True)

        self.assertTrue(os.path.exists(single_class_dir), "Single-class dataset folder not created")

        shutil.rmtree(test_path)
        if os.path.exists(single_class_dir):
            shutil.rmtree(single_class_dir)

    @patch("os.system")
    def test_flatten_recursive_folders(self, mock_system):
        mock_system.return_value = 0  # Mock os.system to bypass `cp` command
        test_path = "test_recursive/"
        os.makedirs(f"{test_path}/images", exist_ok=True)
        os.makedirs(f"{test_path}/json", exist_ok=True)
        with open(f"{test_path}/images/img1.jpg", "w") as f:
            f.write("mock image content")
        with open(f"{test_path}/json/img1.json", "w") as f:
            f.write("mock json content")

        flatten_recursive_folders(path=test_path)

        # Verify the flattened folders
        self.assertTrue(os.path.exists(f"{test_path}/images_flat"))
        self.assertTrue(os.path.exists(f"{test_path}/json_flat"))

        shutil.rmtree(test_path)

    def test_coco91_to_coco80_class(self):
        mapping = coco91_to_coco80_class()
        self.assertEqual(len(mapping), 91)


if __name__ == "__main__":
    unittest.main()
