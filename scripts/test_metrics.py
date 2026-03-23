import sys

import torch

sys.path.append("..")

from deepml.metrics.segmentation import Precision, Recall

if __name__ == "__main__":
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])
    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])

    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)
    precision = Precision(mode="multiclass", num_classes=3, target_class_index=0)
    print(precision(probs, gt))
