"""Triplet loss-based text classifier for occupation classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TripletTextEncoder(nn.Module):
    """BERT-based encoder for triplet learning."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Project BERT hidden size to embedding dimension
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass to get embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            L2-normalized embeddings [batch_size, embedding_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Project to embedding space
        embeddings = self.projection(pooled_output)

        # L2 normalize embeddings for better triplet learning
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class TripletClassifier(nn.Module):
    """Combined triplet learning + classification model."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 38,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        # Shared encoder for triplet learning
        self.encoder = TripletTextEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            dropout=dropout,
            freeze_bert=freeze_bert,
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for classification or embedding extraction.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_embeddings: Whether to return embeddings along with logits

        Returns:
            If return_embeddings=False: Classification logits [batch_size, num_classes]
            If return_embeddings=True: (logits, embeddings)
        """
        # Get embeddings
        embeddings = self.encoder(input_ids, attention_mask)

        # Classification
        logits = self.classifier(embeddings)

        if return_embeddings:
            return logits, embeddings
        return logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get embeddings only."""
        return self.encoder(input_ids, attention_mask)


class TripletMiningStrategy:
    """Strategy for mining triplets from batch data."""

    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def mine_hard_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hard triplets using hardest negative mining.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (anchor_embeddings, positive_embeddings, negative_embeddings)
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for same/different classes
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal

        # For each anchor, find hardest positive and hardest negative
        anchors, positives, negatives = [], [], []

        for i in range(batch_size):
            # Get mask for this anchor
            same_class_mask = labels_equal[i].clone()
            same_class_mask[i] = False  # Exclude self

            diff_class_mask = labels_not_equal[i]

            # Skip if no positives or negatives available
            if not same_class_mask.any() or not diff_class_mask.any():
                continue

            # Find hardest positive (farthest among same class)
            pos_distances = distances[i][same_class_mask]
            hardest_pos_idx = same_class_mask.nonzero(as_tuple=True)[0][pos_distances.argmax()]

            # Find hardest negative (closest among different classes)
            neg_distances = distances[i][diff_class_mask]
            hardest_neg_idx = diff_class_mask.nonzero(as_tuple=True)[0][neg_distances.argmin()]

            anchors.append(embeddings[i])
            positives.append(embeddings[hardest_pos_idx])
            negatives.append(embeddings[hardest_neg_idx])

        if len(anchors) == 0:
            # Return empty tensors if no valid triplets found
            empty = torch.empty(0, embeddings.size(1), device=device)
            return empty, empty, empty

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )

    def mine_random_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_triplets: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine random valid triplets.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            num_triplets: Number of triplets to mine (default: batch_size)

        Returns:
            Tuple of (anchor_embeddings, positive_embeddings, negative_embeddings)
        """
        batch_size = embeddings.size(0)
        if num_triplets is None:
            num_triplets = batch_size

        device = embeddings.device
        anchors, positives, negatives = [], [], []

        # Get unique labels and their indices
        unique_labels = labels.unique()
        label_to_indices = {
            label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels
        }

        for _ in range(num_triplets):
            # Randomly select anchor label that has at least 2 samples
            valid_labels = [label for label, indices in label_to_indices.items() if len(indices) >= 2]
            if len(valid_labels) < 2:
                break

            anchor_label = torch.randint(0, len(valid_labels), (1,)).item()
            anchor_label = valid_labels[anchor_label]

            # Select anchor and positive from same class
            same_class_indices = label_to_indices[anchor_label]
            selected_indices = torch.randperm(len(same_class_indices))[:2]
            anchor_idx = same_class_indices[selected_indices[0]]
            positive_idx = same_class_indices[selected_indices[1]]

            # Select negative from different class
            diff_labels = [label for label in unique_labels.tolist() if label != anchor_label]
            if not diff_labels:
                continue

            negative_label = diff_labels[torch.randint(0, len(diff_labels), (1,)).item()]
            negative_indices = label_to_indices[negative_label]
            negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,)).item()]

            anchors.append(embeddings[anchor_idx])
            positives.append(embeddings[positive_idx])
            negatives.append(embeddings[negative_idx])

        if len(anchors) == 0:
            # Return empty tensors if no valid triplets found
            empty = torch.empty(0, embeddings.size(1), device=device)
            return empty, empty, empty

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )


class CombinedLoss(nn.Module):
    """Combined triplet loss + classification loss."""

    def __init__(
        self,
        triplet_margin: float = 0.5,
        classification_weight: float = 1.0,
        triplet_weight: float = 1.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2.0)
        # Use label smoothing to prevent overconfidence
        # Handle None class_weights to prevent mode collapse
        if class_weights is not None:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.classification_weight = classification_weight
        self.triplet_weight = triplet_weight

        self.mining_strategy = TripletMiningStrategy(margin=triplet_margin)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        use_hard_mining: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            logits: Classification logits [batch_size, num_classes]
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: True labels [batch_size]
            use_hard_mining: Whether to use hard negative mining

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)

        # Triplet loss
        if use_hard_mining:
            anchors, positives, negatives = self.mining_strategy.mine_hard_triplets(embeddings, labels)
        else:
            anchors, positives, negatives = self.mining_strategy.mine_random_triplets(embeddings, labels)

        if anchors.size(0) > 0:
            triplet_loss = self.triplet_loss(anchors, positives, negatives)
        else:
            triplet_loss = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = self.classification_weight * cls_loss + self.triplet_weight * triplet_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "classification_loss": cls_loss.item(),
            "triplet_loss": triplet_loss.item(),
            "num_triplets": anchors.size(0),
        }

        return total_loss, loss_dict
