import torch

class TokenProcessor:
    def get_inserted_content_position(self, content_ids, start_seq=[151653, 198, 9152, 349], end_seq=[304, 419, 2168, 323, 2550]):
        """
        找到两个token序列之间插入内容的位置
        
        输入参数:
            content_ids: 加了内容的token序列 (batch_size, seq_len) 或 (seq_len,)  
            start_seq: 起始定位序列 list，如 [151653, 198, 9152, 349]
            end_seq: 结束定位序列 list，如 [304, 419, 2168, 323, 2550]
        
        返回:
            inserted_lengths: 每个样本中插入内容的长度列表
            inserted_start_indices: 每个样本中插入内容开始位置的列表
        """
        # 处理单个序列的情况

        if len(content_ids.shape) == 1:
            content_ids = content_ids.unsqueeze(0)
        
        inserted_lengths = []
        inserted_start_indices = []
        
        for content_seq in content_ids:
            # 在content_seq中找到start_seq的位置
            start_pos = self._find_subsequence(content_seq, start_seq)
            if start_pos == -1:
                inserted_lengths.append(0)
                inserted_start_indices.append(None)
                continue
                
            # 在content_seq中找到end_seq的位置  
            end_pos = self._find_subsequence(content_seq, end_seq)
            if end_pos == -1:
                inserted_lengths.append(0)
                inserted_start_indices.append(None)
                continue
            
            # 计算插入内容的位置和长度
            insert_start = start_pos + len(start_seq)  # start_seq之后开始
            insert_end = end_pos  # end_seq之前结束
            
            if insert_end > insert_start:
                inserted_length = insert_end - insert_start
                inserted_lengths.append(inserted_length)
                inserted_start_indices.append(insert_start)
            else:
                # 没有插入内容
                inserted_lengths.append(0)
                inserted_start_indices.append(None)
        
        return inserted_lengths, inserted_start_indices

    def _find_subsequence(self, sequence, subsequence):
        """
        在序列中找到子序列的起始位置
        """
        if len(subsequence) == 0:
            return -1
            
        subseq_tensor = torch.tensor(subsequence, device=sequence.device, dtype=sequence.dtype)
        
        # 滑动窗口查找
        for i in range(len(sequence) - len(subsequence) + 1):
            if torch.equal(sequence[i:i+len(subsequence)], subseq_tensor):
                return i
        
        return -1

# 使用示例
def example_usage():
    # 你的数据
    template_ids = torch.tensor([151653, 198, 9152, 349, 304, 419, 2168, 323, 2550, 279, 29749, 13934, 304, 4718, 3561, 13, 151645, 198, 151644, 77091, 198])
    
    content_ids = torch.tensor([                                                                              
         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,                                                                                
         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151653, 198, 9152, 349, 1697, 5622, 2115, 304, 419, 2168, 323, 2550, 279, 29749, 13934, 304, 4718, 3561, 13, 151645, 198, 151644, 77091, 198])
    
    # 修正定位序列 - 分别定位插入点前后的序列
    start_seq = [151653, 198, 9152, 349]  # 插入点之前的序列
    end_seq = [304, 419, 2168, 323, 2550]    # 插入点之后的序列
    
    processor = TokenProcessor()
    lengths, start_indices = processor.get_inserted_content_position(content_ids, start_seq, end_seq)
    
    print(f"插入内容长度: {lengths}")
    print(f"插入内容起始位置: {start_indices}")
    
    # 提取插入的内容
    if start_indices[0] is not None:
        inserted_content = content_ids[start_indices[0]:start_indices[0]+lengths[0]]
        print(f"插入的token内容: {inserted_content.tolist()}")
    
    # 调试信息
    print("\n调试信息:")
    print(f"模版序列: {template_ids.tolist()}")
    print(f"内容序列: {content_ids.tolist()}")
    print(f"查找起始序列: {start_seq}")
    print(f"查找结束序列: {end_seq}")

# 运行示例
if __name__ == "__main__":
    example_usage()
