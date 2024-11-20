import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BatchHardTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5, exclude_bg=False):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.exclude_bg = exclude_bg

    def forward(self, embeddings, labels):
        loss, frac = batch_hard_triplet_loss(embeddings=embeddings, labels=labels, margin=self.margin,
                                             exclude_bg=self.exclude_bg)
        return loss, frac


class BatchAllTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5, exclude_bg=False):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.exclude_bg = exclude_bg

    def forward(self, embeddings, labels):
        loss, frac = batch_all_triplet_loss(embeddings=embeddings, labels=labels, margin=self.margin,
                                            exclude_bg=self.exclude_bg)
        return loss, frac


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Code adapted from: https://github.com/NegatioN/OnlineMiningTripletLoss

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2  + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    Code adapted from: https://github.com/NegatioN/OnlineMiningTripletLoss
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_triplet_mask_without_zeros(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid AND a is not class 0.

    Modified to exclude the background classes being anchor.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    # elimininate possibility of i being class 0! Create NxN boolean matrix with True if not class 0
    i_equal_bg = (labels == 0).unsqueeze(1).unsqueeze(2)

    valid_labels = ~i_equal_k & (i_equal_j & ~i_equal_bg)

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Code adapted from: https://github.com/NegatioN/OnlineMiningTripletLoss
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_positive_triplet_mask_without_zeros(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Function is modified to never return background class as an anchor.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_non_zero = ~((labels.unsqueeze(0) == 0) & (labels.unsqueeze(1) == 0))
    mask = labels_equal & indices_not_equal & labels_non_zero
    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Code adapted from: https://github.com/NegatioN/OnlineMiningTripletLoss
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def batch_hard_triplet_loss(labels, embeddings, margin, exclude_bg, squared=False, verbose=False, scale=True):
    """Build the triplet loss over a batch of embeddings.

    Note: this batch hard triplet loss is adapted to prevent background class 0 from being an anchor!
          Of course, background class CAN be a negative to push away from.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    if exclude_bg:
        mask_anchor_positive = _get_anchor_positive_triplet_mask_without_zeros(labels).float()
    else:
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, max_positive_indxes = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size, 1)
    hardest_negative_dist, max_negative_indxes = anchor_negative_dist.min(1, keepdim=True)

    if verbose:
        retain_idxes = torch.where(hardest_positive_dist != 0)[0]
        for i in range(max_negative_indxes.shape[0]):
            pos_idx = max_positive_indxes[i].item()
            pos_dist = hardest_positive_dist[i]
            neg_idx = max_negative_indxes[i].item()
            neg_dist = hardest_negative_dist[i]
            loss = F.relu(pos_dist - neg_dist + margin).item()
            print(f"Triplet {i}: anchor class {labels[i]} \t "
                  f"Worst positive idx {pos_idx}, class {labels[pos_idx]}, "
                  f"distance {pos_dist.item():.4f} \t"
                  f"Worst negative idx {neg_idx}, class {labels[neg_idx]}, "
                  f"distance {neg_dist.item():.4f} \t loss: {loss:.4f}")

    # removing indexes of distances where hardest_positive_dist is zero, which is the case only if anchor is background
    if exclude_bg:
        retain_idxes = torch.where(hardest_positive_dist != 0)[0]
        hardest_positive_dist = hardest_positive_dist[retain_idxes]
        hardest_negative_dist = hardest_negative_dist[retain_idxes]

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if scale:
        tl = (hardest_positive_dist - hardest_negative_dist) / (hardest_negative_dist.mean() + 1e-16) + margin
    else:
        tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    n_valid_triplets = tl[tl > 1e-16].size(0)
    n_total_triplets = tl.size(0)
    fraction_positive_triplets = torch.Tensor(np.array(n_valid_triplets / (n_total_triplets + 1e-16)))

    if verbose:
        print(f"Batch HARD triplet loss: {triplet_loss:.4f}")
        print(f"fraction of positive triplets: {fraction_positive_triplets:.4f} {type(fraction_positive_triplets)}")
        print('-' * 69)
    return triplet_loss, fraction_positive_triplets


def batch_all_triplet_loss(labels, embeddings, margin, exclude_bg, squared=False, verbose=False, scale=True):
    """Build the triplet loss over a batch of embeddings.

    Note: this batch all triplet loss is adapted to prevent background class 0 from being an anchor!

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    if verbose:
        print('-' * 69)
        print(f"Batch All triplet loss with ALL classes")

    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    if scale:
        triplet_loss = (anchor_positive_dist - anchor_negative_dist) / (anchor_negative_dist.mean() + 1e-16) + margin
    else:
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    if exclude_bg:
        mask = _get_triplet_mask_without_zeros(labels)
    else:
        mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()
    if verbose:
        print(f"Number of valid triplet: {num_valid_triplets}. Number of positive triplets: {num_positive_triplets}")
        count = 0
        for a in range(mask.shape[0]):
            for p in range(mask.shape[1]):
                for n in range(mask.shape[2]):
                    if mask[a, p, n]:
                        pos_dist = anchor_positive_dist[a, p, :]
                        neg_dist = anchor_negative_dist[a, :, n]
                        if scale:
                            loss = F.relu((pos_dist - neg_dist) / (neg_dist.mean() + 1e-16) + margin).item()
                        else:
                            loss = F.relu(pos_dist - neg_dist + margin).item()
                        print(f"Triplet {count}: anchor class {labels[a]} @idx {a} \t "
                              f"positive class {labels[p]} @idx {p} @distance {pos_dist.item():.4f}\t "
                              f"negative class {labels[n]} @idx {n} @distance {neg_dist.item():.4f} \t"
                              f"loss for this triplet: {loss:.4f}")
                        count += 1

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    if verbose:
        print(f"Batch ALL triplet loss: {triplet_loss:.4f}")
        print(f"fraction of positive triplets: {fraction_positive_triplets:.4f} {type(fraction_positive_triplets)}")
        print('-' * 69)

    return triplet_loss, fraction_positive_triplets


class SupConLoss(nn.Module):
    """ Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Code adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, exclude_background=False, verbose=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.exclude_background = exclude_background
        self.verbose = verbose

    def forward(self, features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].

        Returns:
            A loss scalar.
        """
        assert not (self.exclude_background and labels is None), f"You can't exclude background class without " \
                                                                 f"providing labels"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # mask out background-class contrast cases
        if self.exclude_background:
            bg_mask = torch.where(labels == 0, torch.tensor(0), torch.ones_like(labels)).repeat(anchor_count, 1)
            bg_mask = bg_mask.expand(batch_size * anchor_count, batch_size * anchor_count)
            mask = mask * bg_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # get calculate loss on positive pairs:
        if self.exclude_background:
            no_positive_idxes = torch.where(labels == 0, torch.tensor(0), torch.ones_like(labels)).repeat(anchor_count, 1).bool()[:, 0]
            mean_log_prob_pos = mean_log_prob_pos[no_positive_idxes]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        if self.verbose:
            print(f"Mask: {mask}")
            print(f"mean_log_prob_pos: {mean_log_prob_pos}")

        return loss
