import random
import torch
import os
import pickle
from torchvision import datasets, transforms
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.backends import default_backend

# ====================================================================
# MODULE 2 & 3: TRIPARTITE DATA PREPARATION & ENCRYPTION
# ====================================================================

def get_tripartite_subset(local_data, subset_size=32):
    """
    [Bài báo - Section: Local training and tripartite authentication]
    Lấy một tập con D_t^c từ dữ liệu cục bộ để gửi cho đối tác xác thực.
    """
    if len(local_data) < subset_size:
        return local_data
    return random.sample(local_data, subset_size)

def encrypt_tripartite_data(data_subset, partner_rsa_pub_key):
    """
    [Bài báo - Equation 7 & 8]: Mã hóa lai (Hybrid Encryption) cho dữ liệu xác thực.
    - Sử dụng AES đối xứng để mã hóa dữ liệu (m_t^c)[cite: 206].
    - Sử dụng RSA bất đối xứng của đối tác để mã hóa khóa AES (m_tilde_t^c)[cite: 215].
    """
    # 1. Tạo khóa AES (ak) ngẫu nhiên [cite: 205]
    aes_key = os.urandom(32)
    iv = os.urandom(16)
    
    # 2. Serialize dữ liệu và thực hiện Padding PKCS7
    raw_data = pickle.dumps(data_subset)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(raw_data) + padder.finalize()
    
    # 3. Mã hóa dữ liệu bằng AES-CBC (m_t^c) [cite: 206]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    m_t = encryptor.update(padded_data) + encryptor.finalize()

    # 4. Mã hóa khóa AES bằng khóa RSA công khai của đối tác (m_tilde_t^c) [cite: 215]
    m_tilde_t = partner_rsa_pub_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return {
        "m_t": m_t,          # Dữ liệu đã mã hóa
        "m_tilde_t": m_tilde_t, # Khóa AES đã mã hóa
        "iv": iv
    }

def decrypt_tripartite_data(m_t, m_tilde_t, iv, my_rsa_priv_key):
    """
    [Bài báo - Equation 9 & 10]: Giải mã dữ liệu nhận được từ đối tác[cite: 221, 225].
    """
    # 1. Giải mã lấy lại khóa AES (ak) bằng khóa riêng RSA [cite: 221]
    aes_key = my_rsa_priv_key.decrypt(
        m_tilde_t,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # 2. Giải mã dữ liệu m_t bằng khóa AES vừa lấy được [cite: 225]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(m_t) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    raw_data = unpadder.update(padded_data) + unpadder.finalize()
    
    return pickle.loads(raw_data)

# ====================================================================
# DATA LOADING & NON-IID PARTITIONING
# ====================================================================

def get_fashion_mnist(data_path='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Đưa pixel về khoảng [-1, 1]
    ])
    # Fashion-MNIST gồm 60,000 ảnh training và 10,000 ảnh testing [cite: 335]
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

def partition_non_iid(sc, data_list, num_clients=10):
    """
    [Bài báo - Section: Introduction to the datasets]
    Phân phối dữ liệu Fashion-MNIST cho 10 máy khách theo định dạng non-IID[cite: 336].
    """
    # 1. Gom nhóm dữ liệu theo nhãn để tạo tính chất non-IID
    data_list.sort(key=lambda x: x[1])
    
    # 2. Cắt dữ liệu thành các mảnh (shards)
    num_shards = num_clients * 4
    shard_size = len(data_list) // num_shards
    shards = [data_list[i:i + shard_size] for i in range(0, len(data_list), shard_size)]
    
    # 3. Trộn ngẫu nhiên các mảnh
    random.seed(42)
    random.shuffle(shards)
    
    flat_data = []
    for i in range(num_clients):
        local_data = shards[4*i] + shards[4*i + 1] + shards[4*i + 2] + shards[4*i + 3]
        random.shuffle(local_data) 
        flat_data.extend(local_data)
        
    return sc.parallelize(flat_data, numSlices=num_clients)