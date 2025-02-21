import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import boxes as box_ops

from src.data.dataset import get_base_data_loaders
from src.utils import set_seed


def postprocess_detections(
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]],
        box_coder,
        nms_thresh,
        detections_per_img
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_scores = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list,
                                          image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: detections_per_img]
        single_image_scores = []
        for n_keep in keep:
            which_box = n_keep // 10
            same_box_keep = range(10 * which_box, 10 * (which_box + 1))
            single_image_scores.append(scores[same_box_keep])

        all_scores.append(torch.stack(single_image_scores, dim=0))

    return all_scores


class RCNN():
    @staticmethod
    def predict_on_batch(model, batch, image_paths):
        # # Set the model to evaluation mode
        # model.eval()
        # with torch.no_grad():  # Disable gradient calculation
        #     output = model.forward(batch)
        # return output # 2D
        image_names = [sample_metadata["image_name"] for sample_metadata in
                       image_paths]
        model.eval()

        original_image_sizes: list[tuple[int, int]] = []
        for img in batch:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(batch)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        with torch.no_grad():
            features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training = True
        # model.roi_heads.training=True

        #####proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in
                                 num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(),
                                               anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness,
                                                       images.image_sizes,
                                                       num_anchors_per_level)

        # assert targets is not None

        #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        # proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
        #     proposals, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals,
                                                    image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(
            box_features)

        # boxes, scores, labels = model.roi_heads.postprocess_detections(
        #     class_logits, box_regression, proposals, image_shapes)
        # our new code here
        scores = postprocess_detections(
            class_logits, box_regression, proposals, image_shapes,
            box_coder=model.roi_heads.box_coder,
            nms_thresh=0.5,
            detections_per_img=100
        )
        return list(zip(scores, image_names))

    @staticmethod
    def train_one_epoch(
            model: FastRCNNPredictor,
            train_data_loader: DataLoader,
            device: str,
            optimizer: Optimizer,
            logger
    ) -> float:
        """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            train_data_loader (DataLoader): DataLoader for the training data.
            device (str): Device to run the training on ('cpu' or 'cuda').
            optimizer (Optimizer): Optimizer for the training.

        Returns:
            float: Average training loss for the epoch.
        """
        model.train()
        running_train_loss = 0.0
        len_train_loader = len(train_data_loader)

        for i, (imgs, annotations, _) in enumerate(train_data_loader):
            logger.info(f'Batch {i}/{len_train_loader} ')

            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in
                           annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_train_loss += losses.item()

        avg_train_loss = running_train_loss / len_train_loader
        return avg_train_loss

    @staticmethod
    def eval_forward(model, images: list[torch.Tensor],
                     targets: list[dict[str, torch.Tensor]] | None) -> tuple[
        dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:

        """
        Args:
            model
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        model.eval()

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training = True
        # model.roi_heads.training=True

        #####proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in
                                 num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(),
                                               anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness,
                                                       images.image_sizes,
                                                       num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors,
                                                                       targets)
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes,
                                                        anchors)
        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
            proposals, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals,
                                                    image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(
            box_features)

        result: list[dict[str, torch.Tensor]] = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits,
                                                      box_regression, labels,
                                                      regression_targets)
        detector_losses = {"loss_classifier": loss_classifier,
                           "loss_box_reg": loss_box_reg}
        boxes, scores, labels = model.roi_heads.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = model.transform.postprocess(detections, images.image_sizes,
                                                 original_image_sizes)  # type: ignore[operator]
        model.rpn.training = False
        model.roi_heads.training = False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections

    @staticmethod
    def validate_one_epoch(
            model: FastRCNNPredictor,
            val_data_loader: DataLoader,
            device: str
    ) -> float:
        """
        Validate the model for one epoch.

        Args:
            model (torch.nn.Module): The model to validate.
            val_data_loader (DataLoader): DataLoader for the validation data.
            device (str): Device to run the validation on ('cpu' or 'cuda').

        Returns:
            float: Average validation loss for the epoch.
        """
        model.eval()
        running_val_loss = 0.0
        len_val_loader = len(val_data_loader)

        with torch.no_grad():
            for imgs, annotations, _ in val_data_loader:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in
                               annotations]
                loss_dict, _ = RCNN.eval_forward(model, imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                running_val_loss += losses.item()

        avg_val_loss = running_val_loss / len_val_loader
        return avg_val_loss

    @staticmethod
    def evaluate_on_seperate_dataloader(model,  # type: FastRCNNPredictor
                                        val_data_loader,  # type: DataLoader
                                        device: str) -> dict:
        """
        Evaluate the model on a separate validation DataLoader and compute detection metrics,
        including mean Average Precision (mAP) at various IoU thresholds.

        This function performs the following steps:
            1. Iterates over the validation data.
            2. For each image:
               - Extracts its ground truth annotations and image size.
               - Runs the model in evaluation mode to obtain predicted boxes, labels, and scores.
            3. Converts both the ground truth and detections to a COCO-style dictionary.
            4. Uses pycocotools' COCOeval to compute detection metrics.

        Args:
            model (FastRCNNPredictor): The Faster R-CNN model (or similar) to evaluate.
            val_data_loader (DataLoader): DataLoader for the validation data. Each batch is expected to
                return a tuple (images, annotations, _). The `annotations` are expected to be a list of
                dictionaries with keys like "boxes" and "labels".
            device (str): Device to run evaluation on (e.g. "cpu" or "cuda").

        Returns:
            dict: A dictionary containing detection metrics (e.g. mAP, mAP50, mAP75, etc.).
        """
        model.eval()

        all_gt_annotations = []  # list of ground truth annotations in COCO format
        all_dt_annotations = []  # list of detected (predicted) annotations in COCO format
        images_info = []  # list of image info dictionaries (id, width, height)
        ann_id = 1  # unique annotation id counter
        global_img_id = 0  # unique image id counter

        # Iterate over the validation DataLoader.
        for imgs, annotations, _ in tqdm.tqdm(val_data_loader):
            batch_size = len(imgs)
            # Process each image in the current batch.
            for i, img in enumerate(imgs):
                # Assume each image is a tensor of shape (C, H, W)
                _, h, w = img.shape
                # Save image info (COCO requires image id, width, height)
                images_info.append(
                    {"id": global_img_id, "width": w, "height": h})
                # Process the ground truth annotations for this image.
                ann = annotations[i]
                for box, label in zip(ann["boxes"], ann["labels"]):
                    x_min, y_min, x_max, y_max = box.tolist()
                    width_box = x_max - x_min
                    height_box = y_max - y_min
                    all_gt_annotations.append({
                        "id": ann_id,
                        "image_id": global_img_id,
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, width_box, height_box],
                        "area": width_box * height_box,
                        "iscrowd": 0,
                    })
                    ann_id += 1
                global_img_id += 1

            # Move the batch images to the designated device.
            imgs = [img.to(device) for img in imgs]
            # Run inference (the model in eval mode returns detections).
            with torch.no_grad():
                outputs = model(imgs)
            # Process the predictions.
            # Note: The order of the outputs matches the order of the input images.
            for i, output in enumerate(outputs):
                # Compute the corresponding image id.
                # (global_img_id has been incremented for the entire dataset so far)
                current_img_id = global_img_id - batch_size + i
                for box, label, score in zip(output["boxes"], output["labels"],
                                             output["scores"]):
                    x_min, y_min, x_max, y_max = box.tolist()
                    width_box = x_max - x_min
                    height_box = y_max - y_min
                    all_dt_annotations.append({
                        "image_id": current_img_id,
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, width_box, height_box],
                        "score": float(score.item())
                    })

        # Construct a COCO-style ground truth dictionary.
        # Determine all unique category ids from the ground truth.
        category_ids = sorted(
            {ann["category_id"] for ann in all_gt_annotations})
        categories = [{"id": cat_id, "name": str(cat_id)} for cat_id in
                      category_ids]
        coco_gt_dict = {
            "images": images_info,
            "annotations": all_gt_annotations,
            "categories": categories
        }
        import json
        import tempfile
        # Write the ground truth dictionary to a temporary JSON file,
        # which is required by the COCO API.
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.json') as f:
            json.dump(coco_gt_dict, f)
            gt_file = f.name

        # Create a COCO object for ground truth and load detections.
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(all_dt_annotations)

        # Run COCO evaluation.
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract common metrics.
        metrics = {
            "mAP": coco_eval.stats[0],  # x
            # mAP averaged over IoU thresholds 0.50:0.95
            "mAP50": coco_eval.stats[1],  # AP at IoU=0.50 x
            "mAP75": coco_eval.stats[2],  # AP at IoU=0.75
        }

        # Clean up the temporary ground truth file.
        os.remove(gt_file)
        return metrics

    @staticmethod
    def training_loop(
            model: FastRCNNPredictor,
            num_epochs: int,
            train_data_loader: DataLoader,
            val_data_loader: DataLoader,
            device: str,
            seed: int,
            results_dir: str,
            logger,
            subset: str,
            patience: int = 5,
            min_delta: float = 0.01,
            exp_name: str = "test",

    ) -> None:
        """
        Train the model with early stopping.

        Args:
            model (torch.nn.Module): The model to train.
            num_epochs (int): Number of epochs to train the model.
            train_data_loader (DataLoader): DataLoader for the training data.
            val_data_loader (DataLoader): DataLoader for the validation data.
            device (str): Device to run the training on ('cpu' or 'cuda').
            optimizer (Optimizer): Optimizer for the training.
            seed (int): Seed value for reproducibility.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Default is 5.
            min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Default is 0.01.

        Returns:
            None
        """
        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                    weight_decay=0.0005)

        # Set seed for reproducibility
        set_seed(seed)
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            avg_train_loss = RCNN.train_one_epoch(model, train_data_loader,
                                                  device, optimizer, logger)
            logger.info(
                f'Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}')

            # Validate for one epoch
            avg_val_loss = RCNN.validate_one_epoch(model, val_data_loader,
                                                   device)
            logger.info(
                f'Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}')

            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                logger.info(
                    f'Validation loss improved. Saving model checkpoint...')
                # Optional: save the model checkpoint here
                torch.save(model.state_dict(), os.path.join(results_dir,
                                                            f'model_checkpoint_epoch_{epoch}.pth'))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        f'Early stopping triggered. No improvement in validation loss for {patience} epochs.')
                    break

        logger.info("Training completed")

    @staticmethod
    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          num_classes + 1)
        return model

    @staticmethod
    def get_model_from_path(model, path: str):
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model



def test_model():
    train_data_dir = r'/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/'

    train_data_loader, _, complement_data_loader, val_data_loader = get_base_data_loaders(
        data_dir=train_data_dir,
        batch_size=16)

    # 2 classes; Only target class or background
    num_classes = 11  # lub 2
    num_epochs = 2
    model = RCNN.get_model_instance_segmentation(num_classes)

    model.eval()
    for batch in train_data_loader:
        result = model.forward(batch[0])
        print(result[0])
        print(result[1])
        print(result)

    # # parameters
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    #
    # RCNN.training_loop(model=model,
    #               num_epochs=num_epochs,
    #               train_data_loader=train_data_loader,
    #               val_data_loader=val_data_loader,
    #               device="cpu",
    #               optimizer=optimizer,
    #               seed=1984)


if __name__ == '__main__':
    test_model()
