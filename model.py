import torch
import torch.nn as nn

from lib.ssn.ssn import dense_ssn_iter_inference, ssn_iter, sparse_ssn_iter

_MPS_DENSE_PIXEL_LIMIT = 512 * 512


def run_ssn_inference(pixel_f, nspix, n_iter,
                      init_spixel_features=None, init_label_map=None):
    """
    Run the inference-time SSN assignment on the most appropriate backend.

    Accelerators use the dense device-local path to avoid copying feature maps
    back to CPU. Warm-started refinement also uses the dense path on CPU
    because the sparse inference implementation does not accept an initial
    centroid state.
    """
    needs_warm_start = (
        init_spixel_features is not None or init_label_map is not None
    )
    pixel_count = int(pixel_f.shape[-1]) * int(pixel_f.shape[-2])
    use_dense = pixel_f.device.type == "cuda"
    if pixel_f.device.type == "mps":
        use_dense = pixel_count <= _MPS_DENSE_PIXEL_LIMIT

    if needs_warm_start:
        use_dense = True

    if use_dense:
        return dense_ssn_iter_inference(
            pixel_f,
            nspix,
            n_iter,
            init_spixel_features=init_spixel_features,
            init_label_map=init_label_map,
        )
    if pixel_f.device.type == "mps":
        return sparse_ssn_iter(pixel_f.float().cpu(), nspix, n_iter)
    return sparse_ssn_iter(pixel_f, nspix, n_iter)


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(5, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64 * 3 + 5, feature_dim - 5, 3, padding=1),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)

        return self.assign_features(pixel_f)

    def assign_features(self, pixel_f, init_spixel_features=None, init_label_map=None):
        return run_ssn_inference(
            pixel_f,
            self.nspix,
            self.n_iter,
            init_spixel_features=init_spixel_features,
            init_label_map=init_label_map,
        )

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(
            s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(
            s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)
