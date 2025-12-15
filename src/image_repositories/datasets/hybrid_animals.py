from pathlib import Path



class HybridAnimalsDataset:

    def __init__(self):

        current_file = Path(__file__).resolve()
        resources_dir = current_file.parents[3] / 'resources'

        self.dataset_dir = resources_dir / "hybrid_animals"
        self.dataset_dir.mkdir(exist_ok=True)


    def get_images(self):

        images = {}
        for file_path in self.dataset_dir.glob("*.jpg"):
            name = file_path.stem
            images[name] = str(file_path)
        return images
