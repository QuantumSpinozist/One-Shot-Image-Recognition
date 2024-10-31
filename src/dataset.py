import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T


def apply_affine_with_prob(image):
    """
    Applies a random affine transformation to the input image with a probability of 50% for each transformation type.

    The transformations include rotation, translation, scaling, and shear. The image is inverted before and after the transformation.

    Parameters:
    - image (torch.Tensor): Input image tensor of shape (channels, height, width).

    Returns:
    - torch.Tensor: Transformed image tensor of the same shape as the input.
    """
    image = 1.0 - image
    w, h = image.shape[1], image.shape[2]

    affine_params = {
        'degrees': (0, 10),    # Rotation range θ
        'translate': (2/w, 2/h), # Translation range tx, ty (as fraction of image size)
        'scale': (0.8, 1.2),     # Scaling range sx, sy
        'shear': (-0.3, 0.3)     # Shear range ρx, ρy
    }

    # Initialize transformation parameters with no transformation
    degrees = 0
    translate = (0, 0)
    scale = (1.0, 1.0)
    shear = (0.0, 0.0)

    prob = 0.5
    
    # Randomly apply transformations with 50% probability
    if random.random() < prob:
        degrees = affine_params['degrees']

    if random.random() < prob:
        translate = affine_params['translate']
    
    if random.random() < prob:
        scale = affine_params['scale']
    
    if random.random() < prob:
        shear = affine_params['shear']
    
    # Apply the affine transformation with the randomly selected parameters
    transform = T.RandomAffine(
        degrees=degrees, 
        translate=translate,
        scale=scale,  # PyTorch requires scale as a tuple
        shear=shear,
    )
    
    transformed_image = transform(image)
    return 1.0 - transformed_image


class OmniglotDataset(Dataset):
    """
    Custom dataset class for loading and processing the Omniglot dataset.

    The dataset supports retrieval of positive and negative image pairs for one-shot learning tasks.

    Attributes:
    - dataset (torchvision.datasets.Omniglot): The underlying Omniglot dataset.
    - max_classes_idx (int): Maximum index of the classes in the dataset.
    """

    def __init__(self, root, train=True, download=False):
        """
        Initializes the Omniglot dataset.

        Parameters:
        - root (str): Root directory of the dataset.
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - download (bool): If True, downloads the dataset if it is not available.
        """
        super(OmniglotDataset, self).__init__()
        self.dataset = datasets.Omniglot(root=root, background=train, download=download, transform=T.ToTensor())
        self.max_classes_idx = self.dataset[-1][1]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a pair of images and their corresponding label.

        The method ensures that positive pairs are from the same class and negative pairs are from different classes.

        Parameters:
        - index (int): Index of the requested sample.

        Returns:
        - tuple: (image_1, image_2, target) where image_1 and image_2 are the image tensors and target is the label (1 for positive pair, 0 for negative pair).
        """
        # Pick a random class for the first image
        selected_class = random.randint(0, self.max_classes_idx)
        index_1 = selected_class * 20 + random.randint(0, 19)

        # Get the first image
        image_1, _ = self.dataset[index_1]

        # Same class (positive example)
        if index % 2 == 0:
            index_2 = selected_class * 20 + random.randint(0, 19)
            while index_2 == index_1:
                index_2 = selected_class * 20 + random.randint(0, 19)
            image_2, _ = self.dataset[index_2]
            target = torch.tensor(1, dtype=torch.float)

        # Different class (negative example)
        else:
            other_selected_class = random.randint(0, self.max_classes_idx)
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, self.max_classes_idx)
            index_2 = other_selected_class * 20 + random.randint(0, 19)
            image_2, _ = self.dataset[index_2]
            target = torch.tensor(0, dtype=torch.float)

        # Apply transformation with 8 out of 9 probability
        image_1 = apply_affine_with_prob(image_1) if random.random() > 1/9 else image_1
        image_2 = apply_affine_with_prob(image_2) if random.random() > 1/9 else image_2
        return image_1, image_2, target


def find_first_last_indices(entries, prefix):
    """
    Finds the first and last indices of entries that start with a given prefix in a case-insensitive manner.

    Parameters:
    - entries (list of str): List of entries to search through.
    - prefix (str): Prefix to search for.

    Returns:
    - tuple: (first_index, last_index) of the entries matching the prefix.
    """
    # Convert the prefix to lowercase to match case-insensitively
    prefix_lower = prefix.lower() + "/"

    # Find the first index where the entry starts with the lowercase prefix
    first_index = next(i for i, entry in enumerate(entries) if entry.lower().startswith(prefix_lower))

    # Find the last index where the entry starts with the lowercase prefix
    last_index = len(entries) - 1 - next(i for i, entry in enumerate(reversed(entries)) if entry.lower().startswith(prefix_lower))

    return first_index, last_index


class OmniglotDatasetOneShot(Dataset):
    """
    Custom dataset class for loading and processing the Omniglot dataset for one-shot learning.

    The dataset allows selecting a specific alphabet and retrieves a limited number of classes.

    Attributes:
    - dataset (torchvision.datasets.Omniglot): The underlying Omniglot dataset.
    - max_classes_idx (int): Maximum index of the classes in the dataset.
    - alphabet (str or None): Specific alphabet to load; if None, loads all classes.
    - N (int): Number of classes to sample.
    """

    def __init__(self, root, train=False, download=False, alphabet=None, N=20):
        """
        Initializes the Omniglot dataset for one-shot learning.

        Parameters:
        - root (str): Root directory of the dataset.
        - train (bool): If True, loads the training set; otherwise, loads the test set.
        - download (bool): If True, downloads the dataset if it is not available.
        - alphabet (str or None): Specific alphabet to load; if None, loads all classes.
        - N (int): Number of classes to sample.
        """
        super(OmniglotDatasetOneShot, self).__init__()
        self.dataset = datasets.Omniglot(root=root, background=train, download=download, transform=T.ToTensor())
        self.max_classes_idx = self.dataset[-1][1]
        self.alphabet = alphabet
        self.N = N
        self.set_classes_indices()
        self.set_class_examples()

    def set_classes_indices(self):
        """Sets the indices of the classes based on the specified alphabet."""
        if self.alphabet is None:
            self.classes_idx_a = 0
            self.classes_idx_b = self.max_classes_idx
        else:
            a, b = find_first_last_indices(list(self.dataset._characters), self.alphabet)
            self.classes_idx_a = a
            self.classes_idx_b = b

    def set_class_examples(self):
        """Sets the examples for the classes based on the defined indices."""
        b = self.classes_idx_a + min(self.N, self.classes_idx_b - self.classes_idx_a)
        self.class_examples = [self.dataset[20 * class_idx] for class_idx in range(self.classes_idx_a, b + 1)]

    def update_alphabet(self, alphabet=None, N=20):
        """
        Updates the selected alphabet and number of classes.

        Parameters:
        - alphabet (str or None): New alphabet to load; if None, loads all classes.
        - N (int): New number of classes to sample.
        """
        self.alphabet = alphabet
        self.N = N
        self.set_classes_indices()
        self.set_class_examples()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a random image from the selected class for one-shot learning.

        Parameters:
        - index (int): Index of the requested sample.

        Returns:
        - tuple: (image, selected_class) where image is the image tensor and selected_class is the class index.
        """
        r = min(self.N, self.classes_idx_b - self.classes_idx_a)
        b = self.classes_idx_a + r

        # Pick a random class for the first image
        selected_class = random.randint(self.classes_idx_a, b)
        index = selected_class * 20 + random.randint(1, r)

        # Get the first image
        image, _ = self.dataset[index]
        return image, selected_class
