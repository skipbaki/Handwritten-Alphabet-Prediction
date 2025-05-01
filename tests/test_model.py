import pytest
from keras.models import Sequential
from main.model import create_model

def test_model_creation():
    model = create_model()
    # Basic structure checks
    assert isinstance(model, Sequential)
    assert len(model.layers) == 11  # Updated for new architecture
    # Input shape verification
    assert model.input_shape == (None, 28, 28, 1)
    # Output layer check
    output_layer = model.layers[-1]
    assert output_layer.units == 26
    assert output_layer.activation.__name__ == 'softmax'
def test_model_compilation():
    model = create_model()
    # Verify optimizer
    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert model.optimizer.learning_rate == pytest.approx(0.0001)
    # Verify loss and metrics
    assert model.loss == 'categorical_crossentropy'
    assert 'accuracy' in model.metrics_names
