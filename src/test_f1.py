import torch
from sklearn.metrics import f1_score
predictions = torch.tensor([[0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]]).to(dtype=torch.int)
labels = torch.tensor([[0., 1., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]]).to(dtype=torch.int)


tp = ((predictions[:, 1] == 1) & (labels[:, 1] == 1)).sum()
tn = ((labels[:, 1] == 1) & (predictions[:, 1] != 1)).sum()
fp = ((predictions[:, 1] == 1) & (labels[:, 1] != 1)).sum()
precision = tp / (tp + fp + 1e-5)
recall = tp / (tp + tn + 1e-5)
f1 = 2 * precision * recall / (precision + recall + 1e-5)


pred_labels = predictions[:, 1:].argmax(dim=1)
true_labels = labels[:, 1:].argmax(dim=1)
print(f1_score(pred_labels, true_labels, average='micro'))


