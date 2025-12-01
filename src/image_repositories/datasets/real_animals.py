
import kagglehub
from pathlib import Path
import zipfile


class RealAnimalsDataset:

    def __init__(self):

        current_file = Path(__file__).resolve()
        resources_dir = current_file.parents[3] / 'resources'

        self.dataset_dir = resources_dir / "real_animals"
        self.dataset_dir.mkdir(exist_ok=True)

        self.zip_path = self.dataset_dir / "real_animals.zip"

        if any(self.dataset_dir.iterdir()):
            return

        zip_path = Path(kagglehub.dataset_download(
            "iamsouravbanerjee/animal-image-dataset-90-different-animals",
            force_download=True
        ))

        target_zip = self.dataset_dir / zip_path.name
        zip_path.rename(target_zip)

        with zipfile.ZipFile(target_zip, 'r') as z:
            z.extractall(self.dataset_dir)

        target_zip.unlink()

    def get_images(self, indexes_of_images):

        animal_classes_path = self.dataset_dir / '5' / 'animals' / 'animals'

        animal_folders = [f for f in animal_classes_path.iterdir() if f.is_dir()]

        images_dict = {}

        for idx in indexes_of_images:

            folder_index = idx // 60
            animal_folder = animal_folders[folder_index]

            image_files = [f for f in animal_folder.iterdir() if f.is_file()]

            image_index = idx % 60
            image_file = image_files[image_index]

            animal_name = animal_folder.name
            images_dict[animal_name] = str(image_file)

        return images_dict
