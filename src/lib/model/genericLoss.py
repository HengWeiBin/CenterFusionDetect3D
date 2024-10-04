import torch
import numpy as np

from model.losses import (
    FastFocalLoss,
    RegWeightedL1Loss,
    BinRotLoss,
    WeightedBCELoss,
    UncertaintyDepthLoss,
    DecoupledLoss,
    Bbox2DLoss,
    Bbox3DLoss,
)
from model.utils import sigmoidDepth
from utils.image import getAffineTransform


class GenericLoss(torch.nn.Module):
    def __init__(self, config, num_classes):
        super(GenericLoss, self).__init__()
        self.focalLoss = FastFocalLoss()
        self.L1Loss = RegWeightedL1Loss()
        if {f"rotation{i if i else ''}" for i in range(5)} & set(config.heads):
            self.binRotLoss = BinRotLoss()
        if "nuscenes_att" in config.heads:
            self.bceLoss = WeightedBCELoss()
        self.config = config
        self.num_classes = num_classes
        self.normalize2d = config.MODEL.NORM_2D

        # Losses from ClusterFusion
        if config.TRAIN.UNCERTAINTY_LOSS:
            self.uncertainty = UncertaintyDepthLoss()
        if config.weights["bbox2d"] > 0:
            self.bbox2dLoss = Bbox2DLoss()
        if config.weights["bbox3d"] > 0:
            self.bbox3dLoss = Bbox3DLoss()
        if config.DATASET.DECOUPLE_REP:
            self.decoupledLoss = DecoupledLoss()

    def gatherLayerdata(self, data, layerMask):
        """
        Gather layer data by layer mask

        Args:
            data (Tensor): shape(B, C, [Optional: T])
            layerMask (Tensor): shape(B, C)

        Return:
            (Tensor): shape(data.shape)
        """
        assert data.shape[:2] == layerMask.shape

        while len(data.shape) > len(layerMask.shape):
            layerMask = layerMask.unsqueeze(-1)
        layerMask = layerMask.expand_as(data)

        return data.where(layerMask, torch.zeros_like(data))

    def forward(self, outputs, batch):
        # Initialize losses
        zero = torch.tensor(0.0).to(outputs[0]["heatmap"].device)
        losses = {head: zero.clone() for head in self.config.heads}
        losses["total"] = zero.clone()
        for loss in ["lidar_depth", "radar_depth", "bbox2d", "bbox3d"]:
            if self.config.weights[loss] > 0:
                losses[loss] = zero.clone()

        # define transform matrix (for 3D bbox loss only)
        if self.config.weights["bbox3d"] > 0:
            transMat = getAffineTransform(
                batch["meta"]["center"][0].tolist(),
                batch["meta"]["scale"][0].tolist(),
                0,
                self.config.MODEL.OUTPUT_SIZE[::-1],
                inverse=True,
            ).astype(np.float32)

        # Calculate losses by each layer
        self.build_targets(batch, len(outputs))
        for i, output in enumerate(outputs):

            layerMask = batch["layerMask"][:, i]
            layerBatchClassIds = self.gatherLayerdata(batch["classIds"], layerMask)

            # Calculate center output indices
            centerOutputInt = self.gatherLayerdata(
                batch["target"]["heatCenters"], layerMask
            )
            device = output["heatmap"].device
            outputSize = torch.tensor(self.config.MODEL.OUTPUT_SIZE).to(device)
            layerOutputSize = torch.tensor(output["heatmap"].shape[-2:]).to(device)
            scaleFactor = layerOutputSize / outputSize
            centerLayerInt = (centerOutputInt * scaleFactor).long()
            layerBatchIndices = (
                centerLayerInt[:, :, 1] * layerOutputSize[1] + centerLayerInt[:, :, 0]
            )

            # =============== Class and position loss ===============
            heatmapLoss = self.focalLoss(
                output["heatmap"],
                batch[f"heatmap{i}"],
                layerBatchIndices,
                self.gatherLayerdata(batch["mask"], layerMask),
                layerBatchClassIds,
            )
            losses["heatmap"] += heatmapLoss
            losses["total"] += heatmapLoss * self.config.weights["heatmap"]

            # ==================== Depth Loss ====================
            for depthHead in ["depth", "depth2"]:
                if depthHead not in output:
                    continue

                if depthHead not in losses:
                    losses[depthHead] = zero.clone()

                depthLossArgs = {
                    "output": output[depthHead],
                    "mask": self.gatherLayerdata(
                        batch["mask"].unsqueeze(-1).expand_as(batch["depth"]),
                        layerMask,
                    ),
                    "ind": layerBatchIndices,
                    "target": self.gatherLayerdata(batch["depth"], layerMask),
                }

                if self.training and {depthHead, "uncertainty"} <= set(output.keys()):
                    uncertainty = torch.clamp(output["uncertainty"], min=-10, max=10)
                    depthLoss, uncertaintyLoss = self.uncertainty(
                        **depthLossArgs, uncertainty=uncertainty
                    )
                    losses["total"] += uncertaintyLoss * self.config.weights["depth"]

                elif depthHead in output:
                    depthLoss = self.L1Loss(**depthLossArgs)
                    losses["total"] += depthLoss * self.config.weights["depth"]

                losses[depthHead] += depthLoss

            # ==================== Depth Loss (Map) ====================
            # Get the maximum depth head
            maxDepthHead = None
            for i in range(5, 0, -1):
                idx = "" if i == 1 else str(i)
                if f"depth{idx}" in output:
                    maxDepthHead = f"depth{idx}"
                    break
            assert maxDepthHead, "No depth head found in output"

            # Get depthmap output, useless if lidar/radar depth loss/decode loss is not enabled
            if "depthMap" in output:
                depthMap = sigmoidDepth(output["depthMap"])
            else:
                depthMap = output[maxDepthHead]

            # Calculate lidar depth loss
            if self.config.LOSS_WEIGHTS.LIDAR_DEPTH > 0:
                lidar_pc = batch["pc_lidar"].permute(0, 2, 1).contiguous()
                # (B, 3, N) -> (B, N, 3)
                lidarPcLayerMask = lidar_pc > 0
                lidarPcCenterLayerInt = (lidar_pc[..., :2] * scaleFactor).long()
                lidarPcLayerBatchIndices = (
                    lidarPcCenterLayerInt[..., 1] * layerOutputSize[1]
                    + lidarPcCenterLayerInt[..., 0]
                )

                lidarDepthLoss = self.L1Loss(
                    depthMap,
                    lidarPcLayerMask,
                    lidarPcLayerBatchIndices,
                    lidar_pc[..., 2:],  # lidar depth
                )
                losses["lidar_depth"] += lidarDepthLoss
                losses["total"] += lidarDepthLoss * self.config.weights["lidar_depth"]

            # Calculate radar depth loss
            if self.config.LOSS_WEIGHTS.RADAR_DEPTH > 0:
                radar_pc = batch["pc_2d"].permute(0, 2, 1).contiguous()
                # (B, 3, N) -> (B, N, 3)
                radarPcLayerMask = radar_pc > 0
                radarPcCenterLayerInt = (radar_pc[..., :2] * scaleFactor).long()
                radarPcLayerBatchIndices = (
                    radarPcCenterLayerInt[..., 1] * layerOutputSize[1]
                    + radarPcCenterLayerInt[..., 0]
                )

                radarDepthLoss = self.L1Loss(
                    depthMap,
                    radarPcLayerMask,
                    radarPcLayerBatchIndices,
                    radar_pc[..., 2:],  # radar depth
                )
                losses["radar_depth"] += radarDepthLoss
                losses["total"] += radarDepthLoss * self.config.weights["radar_depth"]

            # ==================== Regression Loss ====================
            regression_heads = [
                "reg",
                "widthHeight",
                "dimension",
                "amodal_offset",
                "velocity",
            ]
            if self.config.DATASET.DECOUPLE_REP and "amodal_offset" in output:
                regression_heads.remove("amodal_offset")
                decoupledLoss = self.decoupledLoss(
                    output["amodal_offset"],
                    self.gatherLayerdata(
                        batch["mask"].unsqueeze(-1).expand_as(batch["amodal_offset"]),
                        layerMask,
                    ),
                    layerBatchIndices,
                    self.gatherLayerdata(batch["amodal_offset"], layerMask),
                    truncMask=self.gatherLayerdata(
                        batch["truncMask"]
                        .unsqueeze(-1)
                        .expand_as(batch["amodal_offset"]),
                        layerMask,
                    ),
                )
                losses["amodal_offset"] += decoupledLoss
                losses["total"] += decoupledLoss * self.config.weights["amodal_offset"]

            for head in regression_heads:
                if head in output:
                    regressionLoss = self.L1Loss(
                        output[head],
                        self.gatherLayerdata(
                            batch["mask"].unsqueeze(-1).expand_as(batch[head]),
                            layerMask,
                        ),
                        layerBatchIndices,
                        self.gatherLayerdata(batch[head], layerMask),
                    )
                    losses[head] += regressionLoss
                    losses["total"] += regressionLoss * self.config.weights[head]

            for rotHead in ["rotation", "rotation2"]:
                if rotHead not in output:
                    continue

                rotationLoss = self.binRotLoss(
                    output[rotHead],
                    self.gatherLayerdata(batch["mask"], layerMask),
                    layerBatchIndices,
                    self.gatherLayerdata(batch["rotbin"], layerMask),
                    self.gatherLayerdata(batch["rotres"], layerMask),
                )
                losses[rotHead] += rotationLoss
                losses["total"] += rotationLoss * self.config.weights[rotHead]

            if "nuscenes_att" in output:
                attLoss = self.bceLoss(
                    output["nuscenes_att"],
                    self.gatherLayerdata(batch["nuscenes_att_mask"], layerMask),
                    layerBatchIndices,
                    self.gatherLayerdata(batch["nuscenes_att"], layerMask),
                )
                losses["nuscenes_att"] += attLoss
                losses["total"] += attLoss * self.config.weights["nuscenes_att"]

            # ============= Decode loss =============
            # 2D Bounding box loss
            if "bbox2d" in losses and {"reg", "widthHeight"} <= set(output.keys()):
                widthHeight = output["widthHeight"]
                if self.normalize2d:
                    widthHeight = widthHeight.sigmoid()
                    widthHeight = widthHeight * outputSize[[1, 0]].view(1, 2, 1, 1)
                bbox2dLoss = self.bbox2dLoss(
                    output["reg"],
                    widthHeight,
                    centerLayerInt,
                    self.gatherLayerdata(batch["target"]["bboxes"], layerMask),
                    layerBatchIndices,
                    layerMask,
                )
                losses["bbox2d"] += bbox2dLoss
                losses["total"] += bbox2dLoss * self.config.weights["bbox2d"]

            # 3D Bounding box loss
            if "bbox3d" in losses and {
                "rotation",
                maxDepthHead,
                "dimension",
                "amodal_offset",
            } <= set(output):
                bbox3dLoss = self.bbox3dLoss(
                    output,
                    centerLayerInt,
                    transMat,
                    batch["calib"],
                    self.gatherLayerdata(batch["target"]["bboxes3d"], layerMask),
                    layerBatchIndices,
                    layerMask,
                    maxDepthHead,
                )
                losses["bbox3d"] += bbox3dLoss
                losses["total"] += bbox3dLoss * self.config.weights["bbox3d"]

        losses["total"] /= len(outputs)
        return losses["total"], losses

    def build_targets(self, batch, nLayers):
        """
        Separate targets into different layers, according to their distance to the object
        Results are stored in batch["heatmaps"] and batch["layerMask"]

        Args:
            batch (dict): batch data
            nLayers (int): number of layers

        Return:
            None
        """
        outputArea = self.config.MODEL.OUTPUT_SIZE[0] * self.config.MODEL.OUTPUT_SIZE[1]
        bboxAreaPercents = (
            torch.prod(batch["widthHeight"], dim=2) / outputArea
        )  # (B, max_objs)
        # sizeThresh = [0, 0.0013, 0.0038, 0.0134] # Q1-Q3
        sizeThresh = [[0, 0.0018, 0.0085][l] for l in range(nLayers)]
        batch_size, max_objs = batch["widthHeight"].shape[:2]
        mask = torch.zeros(
            (batch_size, 0, max_objs), dtype=torch.bool, device=batch["image"].device
        )
        for i in range(nLayers):
            thresh = sizeThresh[i : i + 2]
            if len(thresh) == 1:
                mask_ = (bboxAreaPercents > thresh[0]).squeeze(-1)
            else:
                mask_ = (
                    (bboxAreaPercents > thresh[0]) & (bboxAreaPercents < thresh[1])
                ).squeeze(-1)
            mask = torch.cat((mask, mask_[:, None]), dim=1)

        batch["layerMask"] = mask
