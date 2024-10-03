# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals = None,
    y_normals = None,
    x_features = None,
    y_features = None,
    x_weights = None,
    y_weights = None,
    batch_reduction: Union[str, None] = None,
    point_reduction: Union[str, None] = None,
    norm: int = 2,
    abs_cosine: bool = False,
    norm_features: int = 2,
):
    return_normals = x_normals is not None and y_normals is not None
    return_features = x_features is not None and y_features is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    if x_weights is not None:
        if x_weights.shape[0] != N or x_weights.shape[1] != P1:
            raise ValueError("x_weights must be of shape (N, P1).")
        if not (x_weights >= 0).all():
            raise ValueError("x_weights cannot be negative.")
        if x_weights.sum() == 0.0:
            x_weights = x_weights.view(N, P1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum(2) * x_weights).sum() * 0.0,
                    (x.sum(2) * x_weights).sum() * 0.0,
                )
            return ((x.sum(2) * x_weights) * 0.0, (x.sum(2) * x_weights) * 0.0)

    if y_weights is not None:
        if y_weights.shape[0] != N or y_weights.shape[1] != P2:
            raise ValueError("y_weights must be of shape (N, P2).")
        if not (y_weights >= 0).all():
            raise ValueError("y_weights cannot be negative.")
        if y_weights.sum() == 0.0:
            y_weights = y_weights.view(N, P2)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum(2) * x_weights).sum() * 0.0,
                    (x.sum(2) * x_weights).sum() * 0.0,
                )
            return ((x.sum(2) * x_weights) * 0.0, (x.sum(2) * x_weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0] # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if y_weights is not None:
        if x_weights is None:
            x_weights = torch.ones(N, P1, device=y_weights.device)
        x_weights *= knn_gather(y_weights.view(N, P2, 1), x_nn.idx, y_lengths)[..., 0, :].view(N, P1)

    if x_weights is not None:
        cham_x *= x_weights.view(N, P1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if x_weights is not None:
            cham_norm_x *= x_weights.view(N, P1)

    if return_features:
        # Gather the features using the indices
        x_features_near = knn_gather(y_features, x_nn.idx, y_lengths)[..., 0, :]
        
        if norm_features == 2:
            cham_feat_x = ((x_features - x_features_near)**2).mean(dim=2)
        elif norm_features == 1:
            cham_feat_x = ((x_features - x_features_near).abs()).mean(dim=2)
        else:
            raise # Unsupported norm for the features

        if is_x_heterogeneous:
            cham_feat_x[x_mask] = 0.0
        
        if x_weights is not None:
            cham_feat_x *= x_weights.view(N, P1)

    if point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if return_features:
            cham_feat_x = cham_feat_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped
            if return_features:
                cham_feat_x /= x_lengths_clamped

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if return_features:
                cham_feat_x = cham_feat_x.sum()
            if batch_reduction == "mean":
                div = x_weights.sum() if x_weights is not None else max(N, 1)
                cham_x /= div
                if return_normals:
                    cham_norm_x /= div
                if return_features:
                    cham_feat_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    cham_features = cham_feat_x if return_features else None
    cham_weights = x_weights
    return cham_dist, cham_normals, cham_features, cham_weights


def chamfer_distance(
    x,
    y,
    x_lengths = None,
    y_lengths = None,
    x_normals = None,
    y_normals = None,
    x_features = None,
    y_features = None,
    x_weights = None,
    y_weights = None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        x_weights: Optional FloatTensor of shape (N, P1).
        y_weights: Optional FloatTensor of shape (N, P2).
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x, cham_feat_x, cham_weight_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        None,
        None,
        x_weights,
        y_weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )

    if single_directional:
        return (cham_x, None), (cham_norm_x, None), (cham_feat_x, None), (cham_weight_x, None)

    cham_y, cham_norm_y, cham_feat_y, cham_weight_y = _chamfer_distance_single_direction(
        y,
        x,
        y_lengths,
        x_lengths,
        y_normals,
        x_normals,
        y_features,
        x_features,
        y_weights,
        x_weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )

    return (
        (cham_x, cham_y),
        (cham_norm_x, cham_norm_y),
        (cham_feat_x, cham_feat_y),
        (cham_weight_x, cham_weight_y)
    )