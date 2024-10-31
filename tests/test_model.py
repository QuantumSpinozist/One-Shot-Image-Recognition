import sys
import os
import torch
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.model import SiameseNetwork


# Test class for SiameseNetwork
class TestSiameseNetwork:
    @pytest.fixture
    def setup_model(self):
        """Fixture to set up the SiameseNetwork model."""
        model = SiameseNetwork()
        return model

    def test_forward_output_shape(self, setup_model):
        """Test the output shape of the forward method."""
        model = setup_model
        input1 = torch.randn(1, 1, 105, 105)  # Batch size of 1, 1 channel, 105x105
        input2 = torch.randn(1, 1, 105, 105)  # Same dimensions for second input

        output = model(input1, input2)

        # Output shape should be (1, 1) for binary classification
        assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"

    def test_forward_with_different_input_sizes(self, setup_model):
        """Test if the model raises an error for mismatched input sizes."""
        model = setup_model
        input1 = torch.randn(1, 1, 105, 105)  # Valid input
        input2 = torch.randn(1, 1, 50, 50)    # Invalid input size

        with pytest.raises(RuntimeError):
            model(input1, input2)

    def test_model_initialization(self, setup_model):
        """Test if the model is initialized correctly."""
        model = setup_model

        # Check the first layer weights are initialized
        assert model.convnet[0].weight.mean().item() != 0, "Conv2d layer weights should not be zero initialized."
        assert model.fc[0].weight.mean().item() != 0, "Linear layer weights should not be zero initialized."

    def test_forward_output_range(self, setup_model):
        """Test the output of the model is within the expected range [0, 1]."""
        model = setup_model
        input1 = torch.randn(1, 1, 105, 105)
        input2 = torch.randn(1, 1, 105, 105)

        output = model(input1, input2)

        # Output should be in the range of [0, 1] due to the sigmoid activation
        assert output.item() >= 0.0 and output.item() <= 1.0, f"Output {output.item()} is not in range [0, 1]"

    def test_model_device(self, setup_model):
        """Test if the model can be moved to a GPU and back to CPU."""
        if torch.cuda.is_available():
            model = setup_model.to('cuda')
            assert next(model.parameters()).is_cuda, "Model parameters should be on the GPU."

            model = model.to('cpu')
            assert next(model.parameters()).is_cuda is False, "Model parameters should be on the CPU."
