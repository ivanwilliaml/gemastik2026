lg = torch.tensor([[5., 0.], [0., .1]]); lb = torch.tensor([0, 1])
assert focal_loss()(lg, lb) < F.cross_entropy(lg, lb) and focal_loss(weight=[1., 2.])(lg, lb) > 0
