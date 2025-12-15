import os.path
import random
from pathlib import Path

from src.image_repositories.datasets.hand_drawn_animals import HandDrawnAnimalsDataset
from src.image_repositories.datasets.random_objects import RandomObjectsDataset
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

    def get_hand_drawn_AI_animals(self):
        hand_drawn_animals = HandDrawnAnimalsDataset()

        return hand_drawn_animals.get_images()


    def get_random_objects(self, seed):
        random_objects_dataset = RandomObjectsDataset()

        rng = random.Random(seed)

        random_numbers = rng.sample(range(0, 15000), 50)

        return random_objects_dataset.get_images(random_numbers)

