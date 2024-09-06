from easy_viz_cnn.models import SimpleCNNModel


def test_lenet_feature_map(lenet_model: SimpleCNNModel):
    calc = [size for size in lenet_model.features()]
    real = [
        (32, 32),
        (28, 28, 6),
        (14, 14, 6),
        (10, 10, 16),
        (5, 5, 16),
        (1, 120),
    ]
    assert real == calc
