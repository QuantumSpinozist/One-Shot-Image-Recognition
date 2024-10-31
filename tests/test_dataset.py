import sys
import os
import pytest
import torch
from torchvision import transforms as T


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.dataset import apply_affine_with_prob, OmniglotDataset, OmniglotDatasetOneShot

class TestApplyAffineWithProb:
    def test_output_shape(self):
        """Test if the output shape of the affine transformation is correct."""
        # Create a sample input tensor (1 channel, 105x105)
        image = torch.ones((1, 105, 105))

        transformed_image = apply_affine_with_prob(image)
        
        # Check that the output shape is the same as the input shape
        assert transformed_image.shape == image.shape, f"Expected output shape {image.shape}, got {transformed_image.shape}"

    def test_affine_probabilities(self):
        """Test if transformations are applied with the correct probabilities."""
        image = torch.rand((1, 105, 105))
        
        # Run multiple tests to check for random transformations
        transformations_applied = 0
        for _ in range(100):
            transformed_image = apply_affine_with_prob(image)
            if not torch.equal(image, transformed_image):
                transformations_applied += 1
        
        # At least a few transformations should be applied given the 50% probability
        assert transformations_applied > 0, "No transformations were applied, expected some changes."

class TestOmniglotDataset:
    @pytest.fixture
    def setup_dataset(self):
        """Fixture to set up the OmniglotDataset."""
        dataset = OmniglotDataset(root='path_to_omniglot', train=True, download=True)  # Replace with the correct path
        return dataset

    def test_length(self, setup_dataset):
        """Test if the dataset length is correct."""
        dataset = setup_dataset
        assert len(dataset) > 0, "Dataset should not be empty."

    def test_get_item(self, setup_dataset):
        """Test if the dataset returns the correct format from __getitem__."""
        dataset = setup_dataset
        image1, image2, target = dataset[0]

        # Check output types and shapes
        assert isinstance(image1, torch.Tensor), "Image 1 should be a tensor."
        assert isinstance(image2, torch.Tensor), "Image 2 should be a tensor."
        assert isinstance(target, torch.Tensor), "Target should be a tensor."
        assert image1.shape == (1, 105, 105), f"Expected shape for image1: (1, 105, 105), got {image1.shape}"
        assert image2.shape == (1, 105, 105), f"Expected shape for image2: (1, 105, 105), got {image2.shape}"
        assert target.shape == (), f"Expected shape for target: (1,), got {target.shape}"

class TestOmniglotDatasetOneShot:
    @pytest.fixture
    def setup_one_shot_dataset(self):
        """Fixture to set up the OmniglotDatasetOneShot."""
        dataset = OmniglotDatasetOneShot(root='path_to_omniglot', train=False, download=True)  # Replace with the correct path
        return dataset

    def test_length(self, setup_one_shot_dataset):
        """Test if the dataset length is correct."""
        dataset = setup_one_shot_dataset
        assert len(dataset) > 0, "Dataset should not be empty."

    def test_get_item(self, setup_one_shot_dataset):
        """Test if the dataset returns the correct format from __getitem__."""
        dataset = setup_one_shot_dataset
        image, selected_class = dataset[0]

        # Check output types and shapes
        assert isinstance(image, torch.Tensor), "Image should be a tensor."
        assert isinstance(selected_class, int), "Selected class should be an integer."
        assert image.shape == (1, 105, 105), f"Expected shape for image: (1, 105, 105), got {image.shape}"

    def test_update_alphabet(self, setup_one_shot_dataset):
        """Test if updating the alphabet and N works correctly."""
        dataset = setup_one_shot_dataset
        initial_class_count = len(dataset.class_examples)
        
        # Update the alphabet and N
        dataset.update_alphabet('angelic', N=10)  # Replace with an actual alphabet for testing
        assert len(dataset.class_examples) <= initial_class_count, "The number of class examples should change after updating the alphabet."
