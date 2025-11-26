import os.path
import random
from pathlib import Path

from src.image_repositories.datasets.real_animals import RealAnimalsDataset


class DatasetsController:

    def __init__(self):
        current_file = Path(__file__).resolve()
        project_dir = current_file.parents[2]

        os.makedirs(os.path.join(project_dir, 'resources'), exist_ok=True)

    def get_real_animals(self, seed):
        real_animals_dataset = RealAnimalsDataset()

        rng = random.Random(seed)

        random_numbers = rng.sample(range(0, 5400), 50)

        return real_animals_dataset.get_images(random_numbers)

