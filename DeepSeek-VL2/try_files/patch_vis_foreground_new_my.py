import matplotlib.pyplot as plt 

import matplotlib.patches as patches 
from PIL import Image, ImageOps, ImageDraw, ImageFont 
import torch 
import torchvision.transforms as T 
import numpy as np 
import math 
import os 
from matplotlib import cm 
import matplotlib.colors as mcolors

class SimpleImageProcessingVisualizer: 
    """简化版DeepSeek-VL图像处理可视化（attention可视化改为热力图风格，修复序列映射）"""
    def __init__(self, 
                image_size=392,
                patch_size=14,
                downsample_ratio=2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        
        # 图像变换（未在本示例使用）
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_image(self, image_path: str, num_width_tiles: int, num_height_tiles: int):
        """处理图像并生成切片"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 计算目标分辨率
        target_width = num_width_tiles * self.image_size
        target_height = num_height_tiles * self.image_size
        
        # 创建全局视图 (单个image_size)
        global_view = ImageOps.pad(
            image, 
            (self.image_size, self.image_size),
            color=(128, 128, 128)  # 灰色填充
        )
        
        # 创建局部视图 (按tiles切分)
        local_view = ImageOps.pad(
            image,
            (target_width, target_height),
            color=(128, 128, 128)
        )
        
        # 切分局部视图
        local_patches = []
        patch_positions = []
        patch_coords = []  # 记录patch在grid中的坐标
        
        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                x1 = j * self.image_size
                y1 = i * self.image_size
                x2 = x1 + self.image_size
                y2 = y1 + self.image_size
                
                patch = local_view.crop((x1, y1, x2, y2))
                local_patches.append(patch)
                patch_positions.append((x1, y1, x2, y2))
                patch_coords.append((i, j))  # (row, col)
        
        # 计算token数量
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        # 全局视图tokens: h * (w + 1), +1是每行的特殊token
        global_tokens = h * (w + 1)  # 14*(14+1)=210
        # 分隔符tokens: 1
        separator_tokens = 1
        # 局部视图tokens: (num_height_tiles * h) * (num_width_tiles * w + 1)
        local_tokens = (num_height_tiles * h) * (num_width_tiles * w + 1)
        total_tokens = global_tokens + separator_tokens + local_tokens
        
        return {
            'original_image': image,
            'original_size': original_size,
            'target_size': (target_width, target_height),
            'global_view': global_view,
            'local_view': local_view,
            'local_patches': local_patches,
            'patch_positions': patch_positions,
            'patch_coords': patch_coords,
            'num_width_tiles': num_width_tiles,
            'num_height_tiles': num_height_tiles,
            'token_info': {
                'global_tokens': global_tokens,
                'separator_tokens': separator_tokens,
                'local_tokens': local_tokens,
                'total_tokens': total_tokens,
                'h': h, 'w': w
            }
        }

    def load_attention_data(self, npz_path: str):
        """加载attention数据"""
        data = np.load(npz_path)
        attention = data['attention']
        if attention.ndim > 1:
            attention = attention.reshape(-1)
        topk_idx = data['topk_idx'] if 'topk_idx' in data else None
        return attention, topk_idx

    # ----------------------------
    # Token ID -> 位置映射（保留）
    # ----------------------------
    def token_id_to_position(self, token_id: int, num_width_tiles: int, num_height_tiles: int):
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        global_tokens = h * (w + 1)
        separator_tokens = 1
        
        if token_id < global_tokens:
            # 全局token
            row = token_id // (w + 1)
            col = token_id % (w + 1)
            if col < w:  # 图像token
                return {
                    'type': 'global_image',
                    'patch_type': 'global',
                    'patch_row': 0,
                    'patch_col': 0,
                    'token_row': row,
                    'token_col': col
                }
            else:  # 特殊token
                return {
                    'type': 'global_special',
                    'patch_type': 'global',
                    'patch_row': 0,
                    'patch_col': 0,
                    'token_row': row,
                    'token_col': col
                }
        elif token_id == global_tokens:
            # 分隔符token
            return {'type': 'separator'}
        else:
            # 局部token
            local_token_id = token_id - global_tokens - separator_tokens
            tokens_per_row = num_width_tiles * w + 1
            
            # 计算在局部视图中的行
            local_row = local_token_id // tokens_per_row
            # 计算在该行中的列
            local_col = local_token_id % tokens_per_row
            
            # 计算patch坐标
            patch_row = local_row // h
            patch_col = local_col // w
            
            # 计算patch内坐标
            token_row = local_row % h
            token_col = local_col % w
            
            if local_col < num_width_tiles * w and local_col % w < w:  # 图像token
                return {
                    'type': 'local_image',
                    'patch_type': 'local',
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'token_row': token_row,
                    'token_col': token_col
                }
            else:  # 特殊token
                return {
                    'type': 'local_special',
                    'patch_type': 'local',
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'token_row': token_row,
                    'token_col': local_col
                }
        return None

    # ----------------------------
    # Attention 工具
    # ----------------------------
    def _tokens_per_block(self):
        # 每个块（无特殊token）对应的图像token数
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        return h * w, h, w

    def _build_patch_heatmap(self, attn_vec, h, w, token_pixel_size):
        """
        将长度为 h*w 的 attention 向量变为与 patch 图像同尺寸的矩阵（块状上采样），用于imshow叠加
        """
        if attn_vec is None or len(attn_vec) < h * w:
            return None
        mat = np.array(attn_vec[:h*w]).reshape(h, w)
        # 用kron进行块状上采样到 (h*token_pixel_size, w*token_pixel_size)
        heatmap = np.kron(mat, np.ones((token_pixel_size, token_pixel_size), dtype=np.float32))
        return heatmap

    def _add_attention_grid_overlay(self, ax, image_size, patch_size=14, downsample_ratio=2):
        """
        在热力图上添加网格线（白色）
        """
        effective_patch_size = patch_size * downsample_ratio  # 一般为 28
        width, height = image_size
        
        grid_h = height // effective_patch_size
        grid_w = width // effective_patch_size
        
        # 画网格线（坐标与imshow的extent一致）
        for i in range(grid_w + 1):
            x = i * effective_patch_size
            ax.axvline(x=x, color='white', linewidth=1, alpha=0.8)
        for j in range(grid_h + 1):
            y = j * effective_patch_size
            ax.axhline(y=y, color='white', linewidth=1, alpha=0.8)

    def _split_attention_as_blocks(self, attention_data, num_patches):
        """
        假设attention_data是“按块顺序”拼接：
        [全局块(h*w), patch0(h*w), patch1(h*w), ... patchN(h*w)]
        返回：global_block_vec, [patch0_vec, patch1_vec, ...]
        """
        tokens_per_block, h, w = self._tokens_per_block()
        expected_len = tokens_per_block * (1 + num_patches)
        if len(attention_data) < expected_len:
            return None, None, (tokens_per_block, h, w, expected_len)
        
        global_vec = attention_data[:tokens_per_block]
        patches_vec = []
        offset = tokens_per_block
        for _ in range(num_patches):
            patches_vec.append(attention_data[offset:offset+tokens_per_block])
            offset += tokens_per_block
        return global_vec, patches_vec, (tokens_per_block, h, w, expected_len)

    def _split_attention_from_sequence(self, attention_data, num_width_tiles, num_height_tiles):
        """
        从“完整序列”（包含 global(h*(w+1)) + sep(1) + local((H*h)*(W*w+1))）抽取：
        - global_vec: 长度 h*w（每行忽略最后一个特殊 token）
        - patch_vecs: 每个局部 patch 一个长度 h*w 的向量，顺序与 process_image 生成的 local_patches 一致
        """
        attention_data = np.asarray(attention_data).reshape(-1)
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        global_tokens = h * (w + 1)
        separator_tokens = 1
        tokens_per_local_row = num_width_tiles * w + 1
        total_local_rows = num_height_tiles * h
        expected_total = global_tokens + separator_tokens + total_local_rows * tokens_per_local_row
        
        if len(attention_data) < expected_total:
            return None, None, (expected_total,)
        
        # Global: 取每行前 w 个图像 token
        global_rows = []
        for r in range(h):
            row_start = r * (w + 1)
            row_vals = attention_data[row_start : row_start + w]
            global_rows.append(row_vals)
        global_vec = np.concatenate(global_rows, axis=0)  # 长度 h*w
        
        # Local: 从 separator 之后开始
        base = global_tokens + separator_tokens
        patch_vecs = []
        for pr in range(num_height_tiles):
            for pc in range(num_width_tiles):
                mat_rows = []
                for r in range(h):
                    local_row_index = pr * h + r
                    row_start = base + local_row_index * tokens_per_local_row
                    col_start = row_start + pc * w
                    row_vals = attention_data[col_start : col_start + w]
                    mat_rows.append(row_vals)
                patch_vecs.append(np.concatenate(mat_rows, axis=0))
        
        return global_vec, patch_vecs, None

    def _auto_parse_attention(self, attention_data, num_width_tiles, num_height_tiles, num_patches):
        """
        自动判定 attention 的组织形式：
        - 若长度等于完整序列长度：按 sequence 解析
        - 若长度等于 (1+num_patches)*h*w：按 blocks 解析
        - 否则尝试 sequence（需长度>=序列长度），不满足则返回 None
        """
        tokens_per_block, h, w = self._tokens_per_block()
        global_tokens = h * (w + 1)
        separator_tokens = 1
        total_local_rows = num_height_tiles * h
        tokens_per_local_row = num_width_tiles * w + 1
        expected_seq_len = global_tokens + separator_tokens + total_local_rows * tokens_per_local_row
        expected_blk_len = tokens_per_block * (1 + num_patches)
        
        L = len(attention_data)
        # 精确匹配优先
        if L == expected_seq_len:
            g, p, _ = self._split_attention_from_sequence(attention_data, num_width_tiles, num_height_tiles)
            mode = 'sequence'
            return g, p, mode, expected_seq_len
        if L == expected_blk_len:
            g, p, _ = self._split_attention_as_blocks(attention_data, num_patches)
            mode = 'blocks'
            return g, p, mode, expected_blk_len
        # 尝试 sequence 的最小长度
        if L >= expected_seq_len:
            g, p, _ = self._split_attention_from_sequence(attention_data, num_width_tiles, num_height_tiles)
            if g is not None:
                mode = 'sequence'
                return g, p, mode, expected_seq_len
        # 尝试 blocks 的最小长度
        if L >= expected_blk_len:
            g, p, _ = self._split_attention_as_blocks(attention_data, num_patches)
            if g is not None:
                mode = 'blocks'
                return g, p, mode, expected_blk_len
        
        return None, None, None, max(expected_seq_len, expected_blk_len)

    # ----------------------------
    # 原有：绘制网格与 token 编号（保留，不再用attention渲染填充）
    # ----------------------------
    def add_token_grid_to_patch(self, patch_image, patch_type='local', patch_row=0, patch_col=0, 
                            num_width_tiles=1, num_height_tiles=1, start_token_id=0,
                            attention_data=None, topk_idx=None):
        """
        保留原有网格与编号绘制；attention可视化改为热力图方案，这里不再根据attention填充颜色。
        """
        patch_with_grid = patch_image.copy()
        
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        token_pixel_size = self.image_size // h  # 28
        
        try:
            font = ImageFont.truetype("arial.ttf", size=8)
        except:
            font = ImageFont.load_default()
        
        token_id = start_token_id
        
        if patch_type == 'global':
            # 为了显示特殊token，保持扩展画布逻辑，但不使用attention上色
            extended_width = self.image_size + 60
            extended_image = Image.new('RGB', (extended_width, self.image_size), (128, 128, 128))
            extended_image.paste(patch_image, (0, 0))
            draw = ImageDraw.Draw(extended_image)
            
            # 全局视图：h 行，w+1 列（最后一列是特殊token）
            for i in range(h):  # 14行
                for j in range(w + 1):  # 15列
                    if j < w:  # 图像token
                        x1 = j * token_pixel_size
                        y1 = i * token_pixel_size
                        x2 = x1 + token_pixel_size
                        y2 = y1 + token_pixel_size
                        draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                        
                        text = str(token_id)
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (token_pixel_size - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        draw.text((text_x, text_y), text, fill='white', font=font)
                    else:
                        # 特殊token栏位
                        x1 = w * token_pixel_size + 5
                        y1 = i * token_pixel_size
                        x2 = x1 + 50
                        y2 = y1 + token_pixel_size
                        draw.rectangle([x1, y1, x2-1, y2-1], outline='red', width=2)
                        
                        text = f"SP{token_id}"
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (50 - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        draw.text((text_x, text_y), text, fill='red', font=font)
                    token_id += 1
            return extended_image
                    
        else:  # local patch
            should_show_special = (patch_col == num_width_tiles - 1)
            if should_show_special:
                extended_width = self.image_size + 60
                extended_image = Image.new('RGB', (extended_width, self.image_size), (128, 128, 128))
                extended_image.paste(patch_image, (0, 0))
                draw = ImageDraw.Draw(extended_image)
            else:
                draw = ImageDraw.Draw(patch_with_grid)
            
            tokens_per_row = num_width_tiles * w + 1
            
            for i in range(h):  # 14行
                patch_start_in_row = patch_col * w
                row_start_token = start_token_id + (patch_row * h + i) * tokens_per_row + patch_start_in_row
                
                # 绘制图像token
                for j in range(w):  # 14列图像token
                    x1 = j * token_pixel_size
                    y1 = i * token_pixel_size
                    x2 = x1 + token_pixel_size
                    y2 = y1 + token_pixel_size
                    
                    current_token_id = row_start_token + j
                    draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                    
                    text = str(current_token_id)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x1 + (token_pixel_size - text_w) // 2
                    text_y = y1 + (token_pixel_size - text_h) // 2
                    draw.text((text_x, text_y), text, fill='white', font=font)
                
                # 绘制特殊token（只在最右边的patch中显示）
                if should_show_special:
                    special_token_id = start_token_id + (patch_row * h + i) * tokens_per_row + num_width_tiles * w
                    
                    x1 = w * token_pixel_size + 5
                    y1 = i * token_pixel_size
                    x2 = x1 + 50
                    y2 = y1 + token_pixel_size
                    draw.rectangle([x1, y1, x2-1, y2-1], outline='red', width=2)
                    
                    text = f"SP{special_token_id}"
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x1 + (50 - text_w) // 2
                    text_y = y1 + (token_pixel_size - text_h) // 2
                    draw.text((text_x, text_y), text, fill='red', font=font)
            
            return extended_image if should_show_special else patch_with_grid

    # ----------------------------
    # 主可视化：attention 热力图
    # ----------------------------
    def visualize_processing_with_attention(self, image_path: str, num_width_tiles: int, num_height_tiles: int, 
                                        npz_path: str = None, save_path: str = None, parse_mode: str = 'auto'):
        """
        可视化图像处理过程，attention叠加采用热力图风格（plasma + 全局vmin/vmax + 网格 + colorbar）
        parse_mode: 'auto' | 'sequence' | 'blocks'
        """
        result = self.process_image(image_path, num_width_tiles, num_height_tiles)
        
        # 加载attention数据
        attention_data = None
        topk_idx = None
        if npz_path and os.path.exists(npz_path):
            attention_data, topk_idx = self.load_attention_data(npz_path)
            print(f"Loaded attention data: len(attention)={len(attention_data)}, topk={len(topk_idx) if topk_idx is not None else 0}")
        
        # 拆分 attention：得到每块的 h*w 图像 token 注意力
        num_patches = len(result['local_patches'])
        tokens_per_block, h, w = self._tokens_per_block()
        token_pixel_size = self.image_size // h  # 28
        
        global_block_vec = None
        patch_block_vecs = None
        used_mode = None
        expected_len = None
        
        if attention_data is not None:
            if parse_mode == 'sequence':
                gvec, pvecs, meta = self._split_attention_from_sequence(attention_data, num_width_tiles, num_height_tiles)
                if gvec is None:
                    print(f"[Warn] attention长度({len(attention_data)})不足以按完整序列解析(至少需要 {meta[0]}). 将不叠加热力图，仅显示网格。")
                else:
                    global_block_vec, patch_block_vecs = gvec, pvecs
                    used_mode = 'sequence'
            elif parse_mode == 'blocks':
                gvec, pvecs, meta = self._split_attention_as_blocks(attention_data, num_patches)
                if gvec is None:
                    print(f"[Warn] attention长度({len(attention_data)})不足以按块拼接解析(至少需要 {meta[3]}). 将不叠加热力图，仅显示网格。")
                else:
                    global_block_vec, patch_block_vecs = gvec, pvecs
                    used_mode = 'blocks'
            else:
                gvec, pvecs, mode, exp = self._auto_parse_attention(attention_data, num_width_tiles, num_height_tiles, num_patches)
                if gvec is None:
                    print(f"[Warn] 无法自动解析 attention（len={len(attention_data)}），需要至少 {exp}。将不叠加热力图，仅显示网格。")
                else:
                    global_block_vec, patch_block_vecs = gvec, pvecs
                    used_mode = mode
                    expected_len = exp
                    print(f"Attention parse mode: {used_mode}; expected length={expected_len}")
        
        # 全局颜色范围
        global_min = None
        global_max = None
        if global_block_vec is not None and patch_block_vecs is not None:
            all_vals = np.concatenate([np.array(global_block_vec)] + [np.array(x) for x in patch_block_vecs])
            global_min = float(np.min(all_vals))
            global_max = float(np.max(all_vals))
            print(f"Attention global range: [{global_min:.6f}, {global_max:.6f}]")
        
        # 计算subplot布局：1 原图 + 1 全局 + 1 局部视图 + N patch
        total_panels = 3 + num_patches
        cols = min(6, total_panels)
        rows = math.ceil(total_panels / cols)
        fig = plt.figure(figsize=(cols * 3.2, rows * 3.2))
        
        # 1. 原始图像
        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title(f'Original Image\n{result["original_size"]}', fontsize=12)
        ax1.axis('off')
        
        # 2. 全局视图（热力图 + 网格）
        ax2 = plt.subplot(rows, cols, 2)
        # 统一 extent/origin，确保与热力图对齐
        ax2.imshow(result['global_view'], extent=[0, self.image_size, 0, self.image_size], origin='upper', interpolation='nearest', alpha=0.8)
        
        if global_block_vec is not None:
            heatmap_global = self._build_patch_heatmap(global_block_vec, h, w, token_pixel_size)
            if heatmap_global is not None:
                ax2.imshow(
                    heatmap_global, cmap='plasma', alpha=0.8,
                    vmin=global_min, vmax=global_max,
                    extent=[0, self.image_size, 0, self.image_size],
                    origin='upper', interpolation='nearest'
                )
        # 网格线
        self._add_attention_grid_overlay(ax2, (self.image_size, self.image_size), self.patch_size, self.downsample_ratio)
        
        token_info = result['token_info']
        title = f'Global View {self.image_size}×{self.image_size}\nTokens: 0-{token_info["global_tokens"]-1}'
        if global_block_vec is not None:
            title += f'\nAvg: {np.mean(global_block_vec):.3f} | Range: [{np.min(global_block_vec):.3f}, {np.max(global_block_vec):.3f}]'
        ax2.set_title(title, fontsize=11)
        ax2.axis('off')
        
        # 3. 局部视图（整体切分示意）
        ax3 = plt.subplot(rows, cols, 3)
        ax3.imshow(result['local_view'])
        ax3.set_title(f'Local View {result["target_size"]}\n({num_width_tiles}×{num_height_tiles} tiles)', fontsize=12)
        for pos in result['patch_positions']:
            rect = patches.Rectangle(
                (pos[0], pos[1]), 
                pos[2]-pos[0], pos[3]-pos[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax3.add_patch(rect)
        ax3.axis('off')
        
        # 4+. 每个局部patch（热力图 + 网格）
        for i, patch in enumerate(result['local_patches']):
            if 4 + i > rows * cols:
                break
            ax = plt.subplot(rows, cols, 4 + i)
            # 统一 extent/origin
            ax.imshow(patch, extent=[0, self.image_size, 0, self.image_size], origin='upper', interpolation='nearest', alpha=0.8)
            
            block_attn = None
            if patch_block_vecs is not None and i < len(patch_block_vecs):
                block_attn = patch_block_vecs[i]
                heatmap_patch = self._build_patch_heatmap(block_attn, h, w, token_pixel_size)
                if heatmap_patch is not None:
                    ax.imshow(
                        heatmap_patch, cmap='plasma', alpha=0.8,
                        vmin=global_min, vmax=global_max,
                        extent=[0, self.image_size, 0, self.image_size],
                        origin='upper', interpolation='nearest'
                    )
            
            # 添加网格
            self._add_attention_grid_overlay(ax, (self.image_size, self.image_size), self.patch_size, self.downsample_ratio)
            
            # 标题
            row_c, col_c = result['patch_coords'][i]
            title = f'Patch ({row_c},{col_c}) {self.image_size}×{self.image_size}'
            if block_attn is not None:
                title += f'\nAvg: {np.mean(block_attn):.3f} | Range: [{np.min(block_attn):.3f}, {np.max(block_attn):.3f}]'
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # 统一 colorbar
        if global_block_vec is not None and patch_block_vecs is not None:
            sm = cm.ScalarMappable(cmap='plasma', norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
            cbar = plt.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Attention Value', rotation=270, labelpad=12, fontsize=10)
            plt.subplots_adjust(right=0.9)
        
        plt.tight_layout()
        
        # 添加主标题信息
        main_title = f'Image Processing with Attention Heatmap ({num_width_tiles}×{num_height_tiles} tiles)\n'
        main_title += f'Global: {token_info["global_tokens"]} | Separator: {token_info["separator_tokens"]} | Local: {token_info["local_tokens"]} | Total: {token_info["total_tokens"]} tokens'
        if attention_data is not None and global_block_vec is not None:
            mmode = used_mode if used_mode is not None else 'unknown'
            main_title += f'\nHeatmap: plasma colormap, global vmin/vmax | parse_mode={mmode}'
        fig.suptitle(main_title, fontsize=12, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return result

    def print_attention_analysis(self, npz_path: str, num_width_tiles: int, num_height_tiles: int):
        """分析并打印attention统计信息（修复 topk 的索引关联）"""
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}")
            return
        
        attention_data, topk_idx = self.load_attention_data(npz_path)
        attention_data = np.asarray(attention_data).reshape(-1)
        
        print("=" * 60)
        print("Attention Analysis")
        print("=" * 60)
        
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        global_tokens = h * (w + 1)
        separator_tokens = 1
        
        # 统计不同类型token的attention（基于topk的索引）
        global_count = 0
        local_count = 0
        separator_count = 0
        
        global_attentions = []
        local_attentions = []
        
        if topk_idx is None:
            print("No topk_idx found in npz, skip topk-based analysis.")
            return
        
        for token_id in topk_idx:
            if token_id < 0 or token_id >= len(attention_data):
                continue
            att_val = attention_data[token_id]
            if token_id < global_tokens:
                global_count += 1
                global_attentions.append(att_val)
            elif token_id == global_tokens:
                separator_count += 1
            else:
                local_count += 1
                local_attentions.append(att_val)
        
        print(f"Total selected tokens: {len(topk_idx)}")
        print(f"Global tokens with attention: {global_count}")
        print(f"Local tokens with attention: {local_count}")
        print(f"Separator tokens with attention: {separator_count}")
        
        if global_attentions:
            print(f"\nGlobal token attention stats:")
            print(f"  Mean: {np.mean(global_attentions):.4f}")
            print(f"  Max: {np.max(global_attentions):.4f}")
            print(f"  Min: {np.min(global_attentions):.4f}")
        
        if local_attentions:
            print(f"\nLocal token attention stats:")
            print(f"  Mean: {np.mean(local_attentions):.4f}")
            print(f"  Max: {np.max(local_attentions):.4f}")
            print(f"  Min: {np.min(local_attentions):.4f}")



if __name__ == "__main__": # 创建可视化器 
    visualizer = SimpleImageProcessingVisualizer( image_size=392, patch_size=14, downsample_ratio=2 )
    # 设置路径
    # image_path = "/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg"
    image_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/3948.jpg"
    # image_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/3357.jpg"
    # image_path = '/mnt/public/usr/sunzhichao/DeepSeek-VL2/white_1200_800.png'
    # image_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/1087.jpg"


    npz_path = "/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/layer_fastv_attention_vas_new.npz"  # 替换为实际路径

    width_tiles = 2
    height_tiles = 2

    # 可视化图像处理过程（包含attention）
    result = visualizer.visualize_processing_with_attention(
        image_path, 
        width_tiles, 
        height_tiles,
        npz_path=npz_path,
        save_path=f"/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/16_pe_visualization_with_attention_{width_tiles}x{height_tiles}.png",
        parse_mode='auto'  # 可选：'sequence' 或 'blocks'
    )

    # 打印attention分析
    visualizer.print_attention_analysis(npz_path, width_tiles, height_tiles)