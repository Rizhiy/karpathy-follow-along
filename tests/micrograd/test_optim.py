from __future__ import annotations

from flaky import flaky

from micrograd.nn import MLP, MSE
from micrograd.optim import SGD


@flaky
def test_sgd():
    data = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
    targets = [1, -1, -1, 1]

    model = MLP(3, [4, 4, 1])
    loss_func = MSE()
    sgd = SGD(model.parameters(), lr=0.02)

    def get_loss():
        preds = [model(x)[0] for x in data]
        return loss_func(preds, targets)

    starting_loss = get_loss()

    for _ in range(200):
        loss = get_loss()
        sgd.zero_grad()
        loss.backward()
        sgd.step()
    final_loss = get_loss()
    # This sometimes fails, not sure why.
    assert final_loss < starting_loss
    assert final_loss < 0.05
