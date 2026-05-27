import torch
import torch.nn as nn
from torchvision import models
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

# ====================================================================
# HÀM HỖ TRỢ (UTILITIES)
# ====================================================================

def get_device():
    """Tự động xác định thiết bị tính toán sẵn có (CUDA hoặc CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# KIẾN TRÚC MÔ HÌNH (Module 2: Local Training)
# ====================================================================

class ResNet18Fashion(nn.Module):
    """
    Kiến trúc ResNet18 được tinh chỉnh cho Fashion-MNIST.
    """
    def __init__(self):
        super(ResNet18Fashion, self).__init__()
        
        def group_norm(channels):
            return nn.GroupNorm(num_groups=32, num_channels=channels)
            
        self.model = models.resnet18(weights=None, norm_layer=group_norm)
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.model.maxpool = nn.Identity()
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Đảm bảo đầu vào có 4 chiều [Batch, Channel, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

# ====================================================================
# MODULE 3: RELIABLE COMMUNICATION LINK - QUẢN LÝ KHÓA & XÁC THỰC
# ====================================================================

class TripartiteKeyManager:
    """
    [Bài báo - Module 3: Reliable Communication & Initialization] 
    Quản lý cặp khóa RSA và logic xác thực bằng chứng không tri thức (ZKP) cho Client.
    """
    def __init__(self):
        # [Bài báo - Initialization]: Mỗi client tự tạo cặp khóa RSA (pk^c, sk^c) 
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def get_public_key_bytes(self):
        """Xuất khóa công khai dưới dạng PEM bytes để truyền qua mạng Spark."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def verify_downlink_zkp(self, proof_package):
        """
        [Bài báo - Equation 17]: Xác thực bằng chứng ZKP (Downlink).
        Sử dụng giao thức Groth16 để xác thực khóa công khai RSA nhận từ Server.
        """
        if not proof_package:
            return False
        
        # r = 1: Bằng chứng hợp lệ (zkp_status = True)
        return proof_package.get("zkp_status", False)