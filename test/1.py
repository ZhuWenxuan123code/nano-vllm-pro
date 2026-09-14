import torch
import torch.multiprocessing as mp

def producer(queue, num_samples):
    """生产者：生成数据放入队列"""
    for i in range(num_samples):
        # 生成张量并放入队列
        tensor = torch.randn(3, 224, 224)
        queue.put(tensor)
    queue.put(None)  # 结束信号

def consumer(rank, queue, model):
    """消费者：从队列取数据并处理"""
    while True:
        data = queue.get()
        if data is None:  # 收到结束信号
            break
        # 模拟模型推理
        output = model(data)
        print(f"消费者 {rank} 处理了一个样本，输出形状: {output.shape}")
        # 模拟处理时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None

if __name__ == "__main__":
    # 注意：Queue 在 macOS 上可能有序列化问题，建议用 Linux
    queue = mp.Queue()
    
    # 简单的模型
    model = torch.nn.Linear(224*224*3, 10)
    
    # 启动生产者
    producer_process = mp.Process(target=producer, args=(queue, 10))
    producer_process.start()
    
    # 启动多个消费者
    consumers = []
    for i in range(2):
        p = mp.Process(target=consumer, args=(i, queue, model))
        consumers.append(p)
        p.start()
    
    # 等待所有进程完成
    producer_process.join()
    for p in consumers:
        p.join()