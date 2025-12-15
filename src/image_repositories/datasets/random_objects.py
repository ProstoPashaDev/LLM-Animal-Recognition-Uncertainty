
import zipfile
from pathlib import Path

import kagglehub

import requests


class RandomObjectsDataset:

    def __init__(self):

        current_file = Path(__file__).resolve()
        resources_dir = current_file.parents[3] / "resources"

        self.dataset_dir = resources_dir / "random_objects"
        self.dataset_dir.mkdir(exist_ok=True)

        self.zip_path = self.dataset_dir / "random_objects.zip"

        if any(self.dataset_dir.iterdir()):
            return

        zip_path = Path(kagglehub.dataset_download(
            "udaysankarmukherjee/furniture-image-dataset",
            force_download=True
        ))

        target_zip = self.dataset_dir / zip_path.name
        zip_path.rename(target_zip)


    def get_images(self, indexes_of_images):
        images_dict = {}

        random_objects_classes_path = self.dataset_dir / '1'

        random_objects_folders = [f for f in random_objects_classes_path.iterdir() if f.is_dir()]

        for idx in indexes_of_images:

            folder_index = idx // 3000
            random_object_folder = random_objects_folders[folder_index]

            image_files = [f for f in random_object_folder.iterdir() if f.is_file()]

            image_index = idx % 3000
            image_file = image_files[image_index]

            random_object_name = image_file.name
            images_dict[random_object_name] = str(image_file)

        return images_dict


