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
    """简化版DeepSeek-VL图像处理可视化"""
    
    def __init__(self, 
                 image_size=392,
                 patch_size=14,
                 downsample_ratio=2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        
        # 图像变换
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
        global_tokens = h * (w + 1)  # 14 * 15 = 210
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
        topk_idx = data['topk_idx']
        return attention, topk_idx
    
    def token_id_to_position(self, token_id: int, num_width_tiles: int, num_height_tiles: int):
        """将token ID转换为图像中的位置信息"""
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
    
    def add_token_grid_to_patch(self, patch_image, patch_type='local', patch_row=0, patch_col=0, 
                               num_width_tiles=1, num_height_tiles=1, start_token_id=0,
                               attention_data=None, topk_idx=None):
        """在patch上添加token网格和编号，支持attention可视化"""
        # 复制图像避免修改原图
        patch_with_grid = patch_image.copy()
        draw = ImageDraw.Draw(patch_with_grid)
        
        # 计算每个token在图像中的大小
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        token_pixel_size = self.image_size // h  # 392/14 = 28像素对应一个token
        
        try:
            font = ImageFont.truetype("arial.ttf", size=8)
        except:
            font = ImageFont.load_default()
        
        # 创建attention映射
        attention_map = {}
        if attention_data is not None and topk_idx is not None:
            for i, token_id in enumerate(topk_idx):
                if i < len(attention_data):
                    attention_map[token_id] = attention_data[i]
        
        # 获取attention值的范围用于归一化
        if attention_map:
            min_att = min(attention_map.values())
            max_att = max(attention_map.values())
            att_range = max_att - min_att if max_att > min_att else 1
        
        token_id = start_token_id
        
        if patch_type == 'global':
            # 为了显示特殊token，需要扩展画布
            extended_width = self.image_size + 60
            extended_image = Image.new('RGB', (extended_width, self.image_size), (128, 128, 128))
            extended_image.paste(patch_image, (0, 0))
            draw = ImageDraw.Draw(extended_image)
            
            # 全局视图
            for i in range(h):  # 14行
                for j in range(w + 1):  # 15列
                    if j < w:  # 图像token
                        x1 = j * token_pixel_size
                        y1 = i * token_pixel_size
                        x2 = x1 + token_pixel_size
                        y2 = y1 + token_pixel_size
                        
                        # 检查是否有attention值
                        fill_color = None
                        if token_id in attention_map:
                            # 归一化attention值到0-1
                            norm_att = (attention_map[token_id] - min_att) / att_range
                            # 使用热力图颜色
                            color_rgb = cm.Reds(norm_att)[:3]
                            fill_color = tuple(int(c * 255) for c in color_rgb)
                        
                        # 填充背景色（如果有attention）
                        if fill_color:
                            draw.rectangle([x1, y1, x2-1, y2-1], fill=fill_color, outline='white', width=2)
                        else:
                            draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                        
                        # 添加token编号
                        text = str(token_id)
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (token_pixel_size - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        
                        # 根据背景色选择文字颜色
                        text_color = 'black' if fill_color and sum(fill_color) > 400 else 'white'
                        draw.text((text_x, text_y), text, fill=text_color, font=font)
                        
                        # 如果有attention，显示数值
                        if token_id in attention_map:
                            att_text = f"{attention_map[token_id]:.3f}"
                            att_y = text_y + 10
                            draw.text((text_x, att_y), att_text, fill=text_color, font=font)
                        
                    else:
                        # 特殊token
                        x1 = w * token_pixel_size + 5
                        y1 = i * token_pixel_size
                        x2 = x1 + 50
                        y2 = y1 + token_pixel_size
                        
                        # 检查特殊token的attention
                        fill_color = None
                        if token_id in attention_map:
                            norm_att = (attention_map[token_id] - min_att) / att_range
                            color_rgb = cm.Reds(norm_att)[:3]
                            fill_color = tuple(int(c * 255) for c in color_rgb)
                        
                        if fill_color:
                            draw.rectangle([x1, y1, x2-1, y2-1], fill=fill_color, outline='red', width=2)
                        else:
                            draw.rectangle([x1, y1, x2-1, y2-1], outline='red', width=2)
                        
                        text = f"SP{token_id}"
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (50 - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        
                        text_color = 'black' if fill_color and sum(fill_color) > 400 else 'red'
                        draw.text((text_x, text_y), text, fill=text_color, font=font)
                        
                        if token_id in attention_map:
                            att_text = f"{attention_map[token_id]:.3f}"
                            att_y = text_y + 10
                            draw.text((text_x, att_y), att_text, fill=text_color, font=font)
                    
                    token_id += 1
            
            return extended_image
                    
        else:  # local patch
            tokens_per_row = num_width_tiles * w + 1
            
            for i in range(h):  # 14行
                patch_start_in_row = patch_col * w
                row_start_token = start_token_id + (patch_row * h + i) * tokens_per_row + patch_start_in_row
                
                for j in range(w):  # 14列图像token
                    x1 = j * token_pixel_size
                    y1 = i * token_pixel_size
                    x2 = x1 + token_pixel_size
                    y2 = y1 + token_pixel_size
                    
                    current_token_id = row_start_token + j
                    
                    # 检查是否有attention值
                    fill_color = None
                    if current_token_id in attention_map:
                        norm_att = (attention_map[current_token_id] - min_att) / att_range
                        color_rgb = cm.Reds(norm_att)[:3]
                        fill_color = tuple(int(c * 255) for c in color_rgb)
                    
                    # 填充背景色（如果有attention）
                    if fill_color:
                        draw.rectangle([x1, y1, x2-1, y2-1], fill=fill_color, outline='white', width=2)
                    else:
                        draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                    
                    # 添加token编号
                    text = str(current_token_id)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x1 + (token_pixel_size - text_w) // 2
                    text_y = y1 + (token_pixel_size - text_h) // 2
                    
                    # 根据背景色选择文字颜色
                    text_color = 'black' if fill_color and sum(fill_color) > 400 else 'white'
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
                    
                    # 如果有attention，显示数值
                    if current_token_id in attention_map:
                        att_text = f"{attention_map[current_token_id]:.3f}"
                        att_y = text_y + 10
                        draw.text((text_x, att_y), att_text, fill=text_color, font=font)
        
        return patch_with_grid
    
    def visualize_processing_with_attention(self, image_path: str, num_width_tiles: int, num_height_tiles: int, 
                                          npz_path: str = None, save_path: str = None):
        """可视化图像处理过程，包含attention信息"""
        result = self.process_image(image_path, num_width_tiles, num_height_tiles)
        
        # 加载attention数据
        attention_data = None
        topk_idx = None
        if npz_path and os.path.exists(npz_path):
            attention_data, topk_idx = self.load_attention_data(npz_path)
            print(f"Loaded attention data: {len(topk_idx)} tokens with attention values")
            print(f"Attention range: {attention_data.min():.4f} - {attention_data.max():.4f}")
        
        # 计算subplot布局
        total_patches = len(result['local_patches'])
        cols = min(6, total_patches + 3)
        rows = math.ceil((total_patches + 3) / cols)
        
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        
        # 1. 原始图像
        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title(f'Original Image\n{result["original_size"]}', fontsize=12)
        ax1.axis('off')
        
        # 2. 全局视图（带token网格和attention）
        ax2 = plt.subplot(rows, cols, 2)
        global_with_grid = self.add_token_grid_to_patch(
            result['global_view'], 
            patch_type='global', 
            start_token_id=0,
            attention_data=attention_data,
            topk_idx=topk_idx
        )
        ax2.imshow(global_with_grid)
        token_info = result['token_info']
        title = f'Global View\n{self.image_size}×{self.image_size}\nTokens: 0-{token_info["global_tokens"]-1}'
        if attention_data is not None:
            global_tokens_with_att = sum(1 for tid in topk_idx if tid < token_info["global_tokens"])
            title += f'\nAttention: {global_tokens_with_att} tokens'
        ax2.set_title(title, fontsize=12)
        ax2.axis('off')
        
        # 3. 局部视图（带网格）
        ax3 = plt.subplot(rows, cols, 3)
        ax3.imshow(result['local_view'])
        ax3.set_title(f'Local View\n{result["target_size"]}\n({num_width_tiles}×{num_height_tiles} tiles)', fontsize=12)
        
        # 绘制切分网格
        for pos in result['patch_positions']:
            rect = patches.Rectangle(
                (pos[0], pos[1]), 
                pos[2]-pos[0], pos[3]-pos[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax3.add_patch(rect)
        ax3.axis('off')
        
        # 4+. 显示所有局部切片（带token网格、编号和attention）
        local_start_token = token_info['global_tokens'] + token_info['separator_tokens']
        
        for i, patch in enumerate(result['local_patches']):
            if 4 + i > rows * cols:
                break
            ax = plt.subplot(rows, cols, 4 + i)
            
            # 获取patch在grid中的坐标
            row, col = result['patch_coords'][i]
            
            # 为patch添加token网格和attention
            patch_with_grid = self.add_token_grid_to_patch(
                patch,
                patch_type='local',
                patch_row=row,
                patch_col=col,
                num_width_tiles=num_width_tiles,
                num_height_tiles=num_height_tiles,
                start_token_id=local_start_token,
                attention_data=attention_data,
                topk_idx=topk_idx
            )
            
            ax.imshow(patch_with_grid)
            
            # 计算这个patch的token范围和attention统计
            h, w = token_info['h'], token_info['w']
            tokens_per_row = num_width_tiles * w + 1
            patch_start = local_start_token + row * h * tokens_per_row + col * w
            patch_end = patch_start + h * w - 1
            
            title = f'Patch ({row},{col})\n{self.image_size}×{self.image_size}\nTokens: {patch_start}-{patch_end}'
            if attention_data is not None:
                patch_tokens_with_att = sum(1 for tid in topk_idx if patch_start <= tid <= patch_end)
                if patch_tokens_with_att > 0:
                    title += f'\nAttention: {patch_tokens_with_att} tokens'
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # 添加标题信息
        main_title = f'Image Processing with Attention: {num_width_tiles}×{num_height_tiles} tiles\n'
        main_title += f'Global: {token_info["global_tokens"]} | Separator: {token_info["separator_tokens"]} | Local: {token_info["local_tokens"]} | Total: {token_info["total_tokens"]} tokens'
        
        if attention_data is not None:
            main_title += f'\nAttention visualization: {len(topk_idx)} selected tokens (Red intensity = attention strength)'
        
        fig.suptitle(main_title, fontsize=12, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return result
    
    def print_attention_analysis(self, npz_path: str, num_width_tiles: int, num_height_tiles: int):
        """分析并打印attention统计信息"""
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}")
            return
        
        attention_data, topk_idx = self.load_attention_data(npz_path)
        
        print("=" * 60)
        print("Attention Analysis")
        print("=" * 60)
        
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        global_tokens = h * (w + 1)
        separator_tokens = 1
        
        # 统计不同类型token的attention
        global_count = 0
        local_count = 0
        separator_count = 0
        
        global_attentions = []
        local_attentions = []
        
        for i, token_id in enumerate(topk_idx):
            if i < len(attention_data):
                att_val = attention_data[i]
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


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = SimpleImageProcessingVisualizer(
        image_size=392,
        patch_size=14,
        downsample_ratio=2
    )
    
    # 设置路径
    image_path = "/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg"
    npz_path = "/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/layer1_fastv_attention_vas.npz"  # 替换为实际路径
    
    width_tiles = 1
    height_tiles = 2 

    # 可视化图像处理过程（包含attention）
    result = visualizer.visualize_processing_with_attention(
        image_path, 
        width_tiles, 
        height_tiles,
        npz_path=npz_path,
        save_path=f"/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/visualization_with_attention_{width_tiles}x{height_tiles}.png"
    )
    
    # 打印attention分析
    visualizer.print_attention_analysis(npz_path, width_tiles, height_tiles)
