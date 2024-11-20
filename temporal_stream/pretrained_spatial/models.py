import timm
import torch
import torch.nn as nn
from safetensors.torch import load_model
import torch.nn.functional as F


def create_encoder(args, weights_dir=None):
    model_name = args.model
    if "vit" in model_name:
        model = timm.create_model(model_name, in_chans=args.channels, num_classes=0)
        strict = False
    else:
        model = timm.create_model(model_name, in_chans=args.channels)
        strict = True
    print(f"Loaded model {model_name}")

    if weights_dir is not None:
        load_model(model, f"{weights_dir / model_name}.safetensors", strict=strict)
        print(f"Loaded pretrained weights from {weights_dir / model_name}...")
    else:
        print("Using randomly initialized weights...")

    if "vit" in model_name:
        return model, model.embed_dim
    else:
        layers = torch.nn.Sequential(*list(model.children()))
        try:
            potential_last_layer = layers[-1]
            while not isinstance(potential_last_layer, nn.Linear):
                potential_last_layer = potential_last_layer[-1]
        except TypeError:
            raise TypeError('Can\'t find the linear layer of the model')

        features_dim = potential_last_layer.in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])

        return model, features_dim


class ContrastiveModel(nn.Module):
    """
    adapted from https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/models.py
    """
    def __init__(self, args, weights_dir=None, classifier=False, classes=11):
        super(ContrastiveModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(args, weights_dir)
        self.projection_head = True
        self.embed_dim = args.hidden
        self.classifier = classifier

        # we don't drop first head
        self.first_head = nn.Sequential(
            nn.Linear(self.features_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True))

        if self.classifier:
            self.second_head = nn.Sequential(
                nn.Linear(self.embed_dim, classes),
                nn.Softmax())
        else:
            self.second_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim))
            
    def load_weights_encoder(self, path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded model weights for encoder from {path}")

    def use_projection_head(self, mode):
        self.projection_head = mode

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.first_head(feat)
        if self.classifier:
            if self.projection_head:
                logits = self.second_head(feat)
                return logits
            else:
                return F.normalize(feat, dim=0)
        else:
            if self.projection_head:
                return F.normalize(self.second_head(feat), dim=1)
            else:
                if len(feat.shape) == 1:
                    return F.normalize(feat, dim=0)
                else:
                    return F.normalize(feat, dim=1)
