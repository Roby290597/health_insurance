"""
tests/test_neural_network.py
Unit-Tests für das Modell selbst (unabhängig von der API).
"""

import sys 
import os  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from neural_network import InsuranceRegressor


def test_output_shape():
    """Für n Beispiele mit input_dim Features muss die Ausgabe (n, 1) sein."""
    model = InsuranceRegressor(input_dim=12)
    x = torch.randn(5, 12)

    out = model(x)

    assert out.shape == (5, 1)


def test_output_is_finite():
    """Die Vorhersage darf nicht NaN/Inf sein (z.B. bei falscher Initialisierung)."""
    model = InsuranceRegressor(input_dim=12)
    x = torch.randn(3, 12)

    out = model(x)

    assert torch.isfinite(out).all()


def test_deterministic_with_fixed_seed():
    """Gleiche Seed + gleicher Input -> gleiche Ausgabe (Reproduzierbarkeit)."""
    torch.manual_seed(123)
    model_a = InsuranceRegressor(input_dim=4)

    torch.manual_seed(123)
    model_b = InsuranceRegressor(input_dim=4)

    x = torch.randn(2, 4)
    assert torch.allclose(model_a(x), model_b(x))


print("Alle Tests in test_neural_network.py bestanden.")
