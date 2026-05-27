import random
import torch
import os
import pickle
from torchvision import datasets, transforms
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.backends import default_backend

def get_tripartite_subset(local_data, subset_size=32):
    """
    [Bài báo - Section: Local training and tripartite authentication]
    Lấy một tập con D_t^c từ dữ liệu cục bộ để gửi cho đối tác xác thực.
    """
    if len(local_data) < subset_size:
        return local_data
    return random.sample(local_data, subset_size)

def encrypt_tripartite_data(data_subset, partner_rsa_pub_key):

    aes_key = os.urandom(32)
    iv = os.urandom(16)
    
    raw_data = pickle.dumps(data_subset)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(raw_data) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    m_t = encryptor.update(padded_data) + encryptor.finalize()

    m_tilde_t = partner_rsa_pub_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return {
        "m_t": m_t,      
        "m_tilde_t": m_tilde_t, 
        "iv": iv
    }
    
def get_fashion_mnist(data_path='./data'):
"""    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(data_path, train=False, download=True, transform=test_transform)
    
    train_data = [(img.numpy(), label) for img, label in train_set]
    return train_data, test_set
def partition_non_iid(sc, data_list, num_clients=10, shards_per_client=4):
    """
    [Bài báo - Section: Introduction to the datasets]
    Phân phối dữ liệu Fashion-MNIST cho num_clients máy khách theo định dạng non-IID[cite: 336].
 
    Args:
        sc: SparkContext
        data_list: danh sách (image, label)
        num_clients: số lượng client (mặc định 10)
        shards_per_client: số shard mỗi client nhận (mặc định 4, theo bài báo)
    """
    num_shards = num_clients * shards_per_client
 
    # 1. Validate đầu vào trước khi xử lý
    if len(data_list) < num_shards:
        raise ValueError(
            f"Không đủ dữ liệu: cần ít nhất {num_shards} mẫu "
            f"(num_clients={num_clients} × shards_per_client={shards_per_client}), "
            f"nhưng chỉ có {len(data_list)}."
        )
 
    # 2. Gom nhóm dữ liệu theo nhãn để tạo tính chất non-IID
    data_list = sorted(data_list, key=lambda x: x[1])
 
    shards = []
    base_size, remainder = divmod(len(data_list), num_shards)
    start = 0
    for idx in range(num_shards):
        end = start + base_size + (1 if idx < remainder else 0)
        shards.append(data_list[start:end])
        start = end
 
    assert len(shards) == num_shards, \
        f"Số shard thực tế ({len(shards)}) != num_shards ({num_shards})"
    assert start == len(data_list), \
        f"Còn {len(data_list) - start} mẫu chưa được phân bổ vào shard nào"
 
    random.seed(42)
    random.shuffle(shards)
 
    flat_data = []
    for i in range(num_clients):
        local_data = []
        for j in range(shards_per_client):
            local_data += shards[i * shards_per_client + j]
        random.shuffle(local_data)
        flat_data.extend(local_data)
 
    return sc.parallelize(flat_data, numSlices=num_clients)