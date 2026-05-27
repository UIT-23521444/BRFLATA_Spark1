import os, gc, torch, csv, random, sys, math
from pyspark.sql import SparkSession
from network import ResNet18Fashion, TripartiteKeyManager
from dataset import get_fashion_mnist, partition_non_iid, encrypt_tripartite_data, get_tripartite_subset
from federated_logic import local_train_process
from utils import (
    calculate_euclidean_distance, 
    update_aimd, 
    get_adaptive_pairing, 
    verify_rsa_signature
)
from cryptography.hazmat.primitives import serialization
    # hàm tính ngưỡng d_th 
    # """ vô hiệu hóa nếu thực hiện d_th = 6
def threshold_estimation_worker(partition_iterator, global_weights_br, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    
    data = list(partition_iterator)
    if not data or len(data) == 0: return
    if isinstance(data[0], list): data = data[0]
    
    imgs = torch.stack([torch.from_numpy(d[0]) for d in data])
    labels = torch.tensor([d[1] for d in data])
    
    def run_one_iteration():
        model = ResNet18Fashion().to(device)
        model.load_state_dict(global_weights_br.value)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(imgs, labels), batch_size=64, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        model.train()
        for _ in range(3): 
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                criterion(model(batch_x), batch_y).backward()
                optimizer.step()
        
        w = {k: v.cpu() for k, v in model.state_dict().items()}
        del model
        return w

    w_iter1 = run_one_iteration()
    w_iter2 = run_one_iteration()
    
    dist = calculate_euclidean_distance(w_iter1, w_iter2)
    torch.cuda.empty_cache()
    yield dist
    #"""
def evaluate_model(model, test_set):
    """ Đánh giá độ chính xác của mô hình trên tập kiểm thử """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total

def main():
    # 1. KHỞI TẠO HỆ THỐNG SPARK (CÁCH VIẾT AN TOÀN VỚI COMMENT)
    spark = (SparkSession.builder
        .appName("BRFLATA_ResNet18_Full_Implementation")
        .master("local[2]") 
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate())
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    # Thiết lập thông số thực nghiệm (Table 1)
    num_rounds, num_clients = 60, 10
    BYZANTINE_RATIO = 0.6    
    CURRENT_ATTACK_TYPE = 2 
    base_lr = 0.01
    # d_th = 6.0 # (nếu dùng)
    num_attackers = int(num_clients * BYZANTINE_RATIO)
    
    # --- KHỞI TẠO MODULE 1 & 3 ---
    p_history = [] 
    client_managers = {i: TripartiteKeyManager() for i in range(num_clients)}
    
    # Broadcast Public Keys (dạng bytes) để Server xác thực Uplink
    pk_bytes_dict = {i: mgr.get_public_key_bytes() for i, mgr in client_managers.items()}
    pk_bytes_br = sc.broadcast(pk_bytes_dict)

    # KHẮC PHỤC LỖI PICKLE: Chuyển Private Key sang dạng Bytes để gửi tới Worker
    sk_bytes_dict = {i: mgr.private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ) for i, mgr in client_managers.items()}
    sk_bytes_br = sc.broadcast(sk_bytes_dict)

    # Chuẩn bị dữ liệu Non-IID
    raw_train, test_set = get_fashion_mnist()
    federated_data = partition_non_iid(sc, raw_train, num_clients=num_clients).cache()
    
    # Lấy sẵn tập dữ liệu con (subset) để phục vụ mã hóa qua các vòng
    all_local_subsets = federated_data.mapPartitions(lambda it: [get_tripartite_subset(list(it), 512)]).collect()

    global_model = ResNet18Fashion()
    credibility_table = {i: 0.5 for i in range(num_clients)} 
    best_acc = 0.0
    history = []

    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
     # """ vô hiệu hóa nếu thực hiện d_th = 6
    print("\n[*] Đang thực hiện quá trình ước lượng ngưỡng d_th ...")
    gw_init_br = sc.broadcast(global_model.cpu().state_dict())
    
    distances = federated_data.mapPartitions(
        lambda it: threshold_estimation_worker(it, gw_init_br, base_lr)
    ).collect()
    
    estimated_d_th = math.ceil(max(distances)) if distances else 6.0
    d_th = float(estimated_d_th)
    
    print(f"[*] Các khoảng cách dao động thu được: {[round(d, 4) for d in distances]}")
    print(f"[*] Ngưỡng d_th động được thiết lập: {d_th}")
    gw_init_br.unpersist()
    # """
    print("\n" + "="*65)
    print(f" [BRFLATA ULTIMATE] Attack: {CURRENT_ATTACK_TYPE} | Byzantine Ratio: {BYZANTINE_RATIO*100}% ")
    print("="*65)

    # 2. VÒNG LẶP HUẤN LUYỆN CHÍNH
    for r in range(num_rounds):
        # --- MODULE 1: GHÉP CẶP THÍCH ỨNG ---
        pairing = get_adaptive_pairing(credibility_table, r, num_clients, p_history)
        
        # --- SỬA LỖI DECRYPTION FAILED: ĐỒNG BỘ KHÓA MÃ HÓA ---
        # Logic: i xác thực cho partner_id. Vậy partner_id phải mã hóa bằng Public Key của i.
        # Ở đây chúng ta chuẩn bị gói tin mã hóa ĐÚNG cho từng cặp vừa ghép.
        current_encrypted_packages = {}
        for i in range(num_clients):
            partner_id = pairing[i]
            # Lấy Public Key của người nhận (Verifier i)
            verifier_pub_key = client_managers[i].public_key
            # Người gửi (partner_id) thực hiện mã hóa
            current_encrypted_packages[i] = encrypt_tripartite_data(all_local_subsets[partner_id], verifier_pub_key)
        
        forwarded_br = sc.broadcast(current_encrypted_packages)
        
        current_lr = base_lr

        gw_br = sc.broadcast(global_model.cpu().state_dict())
        
        def run_round(index, it):
            attack = CURRENT_ATTACK_TYPE if index < num_attackers else 0 
            # Client 'index' nhận gói tin dành riêng cho mình (đã mã hóa bằng PK của mình)
            pkg_for_me = forwarded_br.value[index]
            sk_bytes = sk_bytes_br.value[index]
            
            # Thực hiện huấn luyện tại Client
            yield (index, next(local_train_process(it, gw_br, pkg_for_me, sk_bytes, attack, current_lr)))

        try:
            raw_results = federated_data.mapPartitionsWithIndex(run_round).collect()
            results_map = {res[0]: res[1] for res in raw_results}
        except Exception as e:
            print(f"[*] Lỗi tại vòng {r+1}: {e}")
            continue

        # --- MODULE 4: SERVER AGGREGATION & AIMD UPDATE ---
        
        round_t = r + 1  # Số vòng thực tế (1 đến 60)
        lambda_val = 10
        
        # Công thức chuẩn xác của Thành để kích hoạt vòng phục hồi
        is_rehab_round = ((round_t - 1) % lambda_val == 0) and (round_t > 1)
        
        if is_rehab_round:
            print(f"")
        for i in range(num_clients):
            partner_id = pairing[i]
            
            is_valid_sig = verify_rsa_signature(
                pk_bytes_br.value[partner_id], 
                results_map[partner_id]["w"], 
                results_map[partner_id]["sig_w"]
            )
            
            dist = calculate_euclidean_distance(results_map[partner_id]["w"], results_map[i]["w_hat"])
            credibility_table[partner_id] = update_aimd(credibility_table[partner_id], dist, d_th, is_valid_sig,is_rehab=is_rehab_round)

        total_r, new_weights_sum = 0, {}
        for i in range(num_clients):
            r_c = credibility_table[i]
            
            if r_c <= 0.01:
                continue
                
            total_r += r_c
            for k, v in results_map[i]["w"].items():
                new_weights_sum[k] = new_weights_sum.get(k, 0) + v * r_c
                
        # BƯỚC 4.3: Cập nhật lịch sử và Global Model
        p_history.extend([(c, p) for c, p in pairing.items()])
        if len(p_history) > 30: p_history = p_history[-30:]

        agg_state = {k: v / (total_r + 1e-9) for k, v in new_weights_sum.items()}
        global_model.load_state_dict(agg_state)

        acc = evaluate_model(global_model, test_set)
        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), "checkpoints/brflata_ultimate.pth")
        
        print(f"Round {r+1:02d} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | r_C0 (Atk): {credibility_table[0]:.4f}")
        # --- BẢNG THEO DÕI ĐIỂM TIN CẬY ---
        print("Danh sách điểm tin cậy r_c:")
        for client_id, score in credibility_table.items():
            status = "(Attacker)" if score <= 0.001 else " (Honest)"
            print(f" - Client {client_id}: {score:.4f} {status}")
        print("-" * 50)
        with open('client_credibility_history2.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Nếu là vòng đầu tiên, ghi header
            if r == 0:
                header = ['Round'] + [f'Client_{i}' for i in range(num_clients)]
                writer.writerow(header)
            
            # Ghi điểm r_c của vòng hiện tại
            row = [r + 1] + [credibility_table[i] for i in range(num_clients)]
            writer.writerow(row)
        history.append([r+1, acc, credibility_table[0]])

        gw_br.unpersist(); forwarded_br.unpersist()
        gc.collect(); torch.cuda.empty_cache()

    # Lưu log cho biểu đồ
    log_file = f'log_brflata_atk{CURRENT_ATTACK_TYPE}_ratio{BYZANTINE_RATIO}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Accuracy', 'Credibility_C0'])
        writer.writerows(history)
    
    spark.stop()

if __name__ == "__main__":
    main()