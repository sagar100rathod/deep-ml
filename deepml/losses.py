import torch
import torch.nn.functional as F


class JaccardLoss(torch.nn.Module):
    """Jaccard Loss (Intersection over Union) for segmentation tasks.

    Computes 1 - IoU as a differentiable loss function for both binary
    and multiclass segmentation.

    Attributes:
        activation: Activation function applied to output logits.
            Softmax2d for multiclass, Sigmoid for binary.
    """

    def __init__(self, is_multiclass):
        """Initializes JaccardLoss with the appropriate activation.

        Args:
            is_multiclass: If True, uses Softmax2d activation for
                multiclass segmentation. Otherwise, uses Sigmoid for
                binary segmentation.
        """
        super(JaccardLoss, self).__init__()
        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = torch.nn.Sigmoid()

    def forward(self, output, target):
        """Computes the Jaccard loss between predictions and targets.

        Args:
            output: Raw model output logits of shape (N, C, H, W).
            target: Ground truth tensor of the same shape as output.

        Returns:
            Scalar tensor representing 1 - mean(IoU).
        """
        output = self.activation(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)

        jac = (intersection / (union - intersection + 1e-7)).mean()
        return 1 - jac


class RMSELoss(torch.nn.Module):
    """Root Mean Squared Error loss.

    Computes sqrt(MSE + eps) to provide a differentiable RMSE loss
    that avoids numerical instability near zero.

    Attributes:
        mse: Underlying MSELoss module.
        eps: Small epsilon value added before the square root for
            numerical stability.
    """

    def __init__(self, eps=1e-6):
        """Initializes RMSELoss.

        Args:
            eps: Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, output, target):
        """Computes the RMSE loss.

        Args:
            output: Predicted tensor of arbitrary shape.
            target: Ground truth tensor of the same shape as output.

        Returns:
            Scalar tensor representing sqrt(MSE(output, target) + eps).
        """
        return torch.sqrt(self.mse(output, target) + self.eps)


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    """Weighted Binary Cross-Entropy loss with logits.

    Applies separate weights to positive and negative samples in the
    binary cross-entropy computation.

    Attributes:
        w_p: Weight for positive samples.
        w_n: Weight for negative samples.
    """

    def __init__(self, w_p=None, w_n=None):
        """Initializes WeightedBCEWithLogitsLoss.

        Args:
            w_p: Weight applied to the positive class loss term.
                Defaults to None.
            w_n: Weight applied to the negative class loss term.
                Defaults to None.
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.w_p = w_p
        self.w_n = w_n

    def forward(self, logits, labels, epsilon=1e-7):
        """Computes the weighted binary cross-entropy loss.

        Args:
            logits: Raw model output logits of shape (N,) or (N, 1).
            labels: Binary ground truth labels of shape (N,).
            epsilon: Small constant to avoid log(0). Defaults to 1e-7.

        Returns:
            Scalar tensor representing the weighted BCE loss.
        """
        ps = torch.sigmoid(logits.squeeze())
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(
            self.w_n * (1 - labels) * torch.log((1 - ps) + epsilon)
        )
        loss = loss_pos + loss_neg
        return loss


class ContrastiveLoss(torch.nn.Module):
    """Contrastive loss for siamese networks.

    Encourages embeddings of similar pairs to be close together and
    embeddings of dissimilar pairs to be at least margin apart.

    Attributes:
        margin: Minimum distance margin between negative pairs.
        distance_func: Optional custom distance function. If None,
            pairwise Euclidean distance is used.
        label_transform: Optional transformation applied to target labels
            before loss computation.
    """

    def __init__(self, margin=2.0, distance_func=None, label_transform=None):
        """Initializes ContrastiveLoss.

        Args:
            margin: The distance margin between positive and negative
                class. Defaults to 2.0.
            distance_func: Custom distance function to use. If None,
                Euclidean pairwise distance is used. Defaults to None.
            label_transform: Transformation function to apply on the
                target label, e.g., lambda label: label[:, 0].
                Defaults to None.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_func = distance_func
        self.label_transform = label_transform

    def forward(self, embeddings: torch.Tensor, label: torch.Tensor):
        """Computes the contrastive loss for a pair of embeddings.

        Args:
            embeddings: A tuple of two tensors (embeddings1, embeddings2),
                each of shape (N, D) where D is the embedding dimension.
            label: Tensor of shape (N,). A value of 1 indicates a
                positive (similar) pair; 0 indicates a negative
                (dissimilar) pair.

        Returns:
            Scalar tensor representing the mean contrastive loss.
        """
        embeddings1, embeddings2 = embeddings
        distance = (
            self.distance_func(embeddings)
            if self.distance_func
            else F.pairwise_distance(embeddings1, embeddings2)
        )
        label = self.label_transform(label) if self.label_transform else label

        pos = label * torch.pow(distance, 2)
        neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


class AngularPenaltySMLoss(torch.nn.Module):
    """Angular Penalty Softmax Loss for deep face recognition.

    Implements three angular margin-based softmax losses:

    - **ArcFace**: Additive angular margin loss.
      See `ArcFace <https://arxiv.org/abs/1801.07698>`_.
    - **SphereFace**: Multiplicative angular margin loss.
      See `SphereFace <https://arxiv.org/abs/1704.08063>`_.
    - **CosFace**: Additive cosine margin loss.
      See `CosFace <https://arxiv.org/abs/1801.05599>`_.

    Attributes:
        s: Scaling factor for the logits.
        m: Angular or cosine margin penalty.
        loss_type: One of 'arcface', 'sphereface', or 'cosface'.
        in_features: Size of the input feature vector.
        out_features: Number of output classes.
        fc: Fully connected layer mapping input features to class logits
            (without bias).
        eps: Small epsilon for numerical stability in acos clamping.
    """

    def __init__(
        self, in_features, out_features, loss_type="arcface", eps=1e-7, s=None, m=None
    ):
        """Initializes AngularPenaltySMLoss.

        Args:
            in_features: Dimensionality of the input feature embeddings.
            out_features: Number of target classes.
            loss_type: Type of angular penalty loss. Must be one of
                'arcface', 'sphereface', or 'cosface'.
                Defaults to 'arcface'.
            eps: Small constant for numerical stability when clamping
                values for acos. Defaults to 1e-7.
            s: Scaling factor for logits. If None, uses the default for
                the chosen loss type (64.0 for arcface/sphereface, 30.0
                for cosface). Defaults to None.
            m: Margin penalty. If None, uses the default for the chosen
                loss type (0.5 for arcface, 1.35 for sphereface, 0.4
                for cosface). Defaults to None.

        Raises:
            AssertionError: If loss_type is not one of the supported types.
        """
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface"]
        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """Computes the angular penalty softmax loss.

        Args:
            x: Input feature embeddings of shape (N, in_features).
            labels: Ground truth class labels of shape (N,), with values
                in the range [0, out_features).

        Returns:
            Scalar tensor representing the negative mean log probability.

        Raises:
            AssertionError: If input and labels have mismatched batch sizes,
                or if labels contain values outside the valid range.
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(
                        torch.diagonal(wf.transpose(0, 1)[labels]),
                        -1.0 + self.eps,
                        1 - self.eps,
                    )
                )
                + self.m
            )
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(
                        torch.diagonal(wf.transpose(0, 1)[labels]),
                        -1.0 + self.eps,
                        1 - self.eps,
                    )
                )
            )

        excl = torch.cat(
            [
                torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(labels)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
