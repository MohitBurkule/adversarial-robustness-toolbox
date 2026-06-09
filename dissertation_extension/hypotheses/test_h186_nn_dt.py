"""
Pytest tests for h186_nn_as_decision_tree.py
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h186_nn_as_decision_tree import (
    ReLUMLP, get_activation_pattern, get_effective_affine,
    dt_predict, boundary_density, experiment_toy, DEVICE, SEED,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def _make_small_model():
    """Train a tiny 2->8->8->3 ReLU MLP on random data."""
    model = ReLUMLP(2, [8, 8], 3).to(DEVICE)
    # Quick training on random data so weights are non-trivial
    X = torch.randn(200, 2, device=DEVICE)
    Y = torch.randint(0, 3, (200,), device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def test_activation_pattern_shape():
    """Single known input -> pattern is binary string of length = total hidden neurons."""
    model = _make_small_model()
    x = torch.tensor([[1.0, -0.5]], device=DEVICE)
    patterns = get_activation_pattern(model, x)
    assert len(patterns) == 1
    pat = patterns[0]
    total_hidden = 8 + 8  # two hidden layers of size 8
    assert len(pat) == total_hidden, f"Expected {total_hidden}, got {len(pat)}"
    assert set(pat) <= {"0", "1"}, f"Pattern contains non-binary chars: {set(pat)}"


def test_effective_affine_matches_forward():
    """get_effective_affine for a sample produces same logits as model forward."""
    model = _make_small_model()
    x = torch.tensor([[0.7, -1.2]], device=DEVICE)
    pat = get_activation_pattern(model, x)[0]
    W_eff, b_eff = get_effective_affine(model, pat)
    x_np = x[0].cpu().numpy().astype(np.float64)
    logits_dt = W_eff @ x_np + b_eff

    with torch.no_grad():
        logits_mlp = model(x)[0].cpu().numpy().astype(np.float64)

    np.testing.assert_allclose(logits_dt, logits_mlp, atol=1e-5,
                               err_msg="Effective affine logits != model forward logits")


def test_dt_predict_equals_model():
    """On 50 random inputs, dt_predict == model(x).argmax() for all."""
    model = _make_small_model()
    X = torch.randn(50, 2, device=DEVICE)
    dt_preds = dt_predict(model, X)
    with torch.no_grad():
        mlp_preds = model(X).argmax(1).cpu().numpy()
    assert np.all(dt_preds == mlp_preds), (
        f"Mismatch on {(dt_preds != mlp_preds).sum()}/50 samples"
    )


def test_boundary_density_far_from_boundary():
    """Input with large feature values -> boundary_density(eps=0.001) should be very small."""
    model = _make_small_model()
    # Very large inputs: pre-activations will be far from zero
    x = torch.tensor([[100.0, 100.0]], device=DEVICE)
    with torch.no_grad():
        bd = boundary_density(model, x, eps=0.001).item()
    assert bd <= 2, f"Expected very small boundary density, got {bd}"


def test_toy_experiment_equivalence():
    """Run experiment_toy(), verify equivalence_rate >= 0.99."""
    results = experiment_toy()
    assert results["test_equiv"] >= 0.99, (
        f"Test equivalence {results['test_equiv']:.4f} < 0.99"
    )
    assert results["grid_equiv"] >= 0.99, (
        f"Grid equivalence {results['grid_equiv']:.4f} < 0.99"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
