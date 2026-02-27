import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import numpy as np
import math
import os

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
    
    def load_topk_data(self, npz_path: str):
        """加载topk token数据"""
        data = np.load(npz_path)
        topk_idx = data['topk_idx']
        return set(topk_idx.flatten())  # 转换为set便于查找
    
    def add_token_grid_to_patch(self, patch_image, patch_type='local', patch_row=0, patch_col=0, 
                               num_width_tiles=1, num_height_tiles=1, start_token_id=0,
                               selected_tokens=None):
        """在patch上添加token网格和编号，高亮显示选中的token"""
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
        
        # 如果没有传入选中的tokens，设为空集
        if selected_tokens is None:
            selected_tokens = set()
        
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
                        
                        # 检查是否是选中的token
                        is_selected = token_id in selected_tokens
                        
                        if is_selected:
                            # 选中的token用红色背景高亮
                            draw.rectangle([x1, y1, x2-1, y2-1], fill='red', outline='yellow', width=3)
                            text_color = 'white'
                        else:
                            # 普通token用白色边框
                            draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                            text_color = 'white'
                        
                        # 添加token编号
                        text = str(token_id)
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (token_pixel_size - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        
                        draw.text((text_x, text_y), text, fill=text_color, font=font)
                        
                    else:
                        # 特殊token
                        x1 = w * token_pixel_size + 5
                        y1 = i * token_pixel_size
                        x2 = x1 + 50
                        y2 = y1 + token_pixel_size
                        
                        is_selected = token_id in selected_tokens
                        
                        if is_selected:
                            draw.rectangle([x1, y1, x2-1, y2-1], fill='red', outline='yellow', width=3)
                            text_color = 'white'
                        else:
                            draw.rectangle([x1, y1, x2-1, y2-1], outline='red', width=2)
                            text_color = 'red'
                        
                        text = f"SP{token_id}"
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]
                        text_x = x1 + (50 - text_w) // 2
                        text_y = y1 + (token_pixel_size - text_h) // 2
                        
                        draw.text((text_x, text_y), text, fill=text_color, font=font)
                    
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
                    
                    # 检查是否是选中的token
                    is_selected = current_token_id in selected_tokens
                    
                    if is_selected:
                        # 选中的token用红色背景高亮
                        draw.rectangle([x1, y1, x2-1, y2-1], fill='red', outline='yellow', width=3)
                        text_color = 'white'
                    else:
                        # 普通token用白色边框
                        draw.rectangle([x1, y1, x2-1, y2-1], outline='white', width=1)
                        text_color = 'white'
                    
                    # 添加token编号
                    text = str(current_token_id)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x1 + (token_pixel_size - text_w) // 2
                    text_y = y1 + (token_pixel_size - text_h) // 2
                    
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        return patch_with_grid
    
    def visualize_processing_with_selected_tokens(self, image_path: str, num_width_tiles: int, num_height_tiles: int, 
                                                npz_path: str = None, save_path: str = None):
        """可视化图像处理过程，高亮显示选中的token"""
        result = self.process_image(image_path, num_width_tiles, num_height_tiles)
        
        # 加载选中的token
        selected_tokens = set()
        if npz_path and os.path.exists(npz_path):
            selected_tokens = self.load_topk_data(npz_path)
            print(f"Loaded {len(selected_tokens)} selected tokens")
            print(f"Selected token IDs: {sorted(list(selected_tokens))[:10]}...")  # 显示前10个
        
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
        
        # 2. 全局视图（带token网格和选中标记）
        ax2 = plt.subplot(rows, cols, 2)
        global_with_grid = self.add_token_grid_to_patch(
            result['global_view'], 
            patch_type='global', 
            start_token_id=0,
            selected_tokens=selected_tokens
        )
        ax2.imshow(global_with_grid)
        token_info = result['token_info']
        
        # 统计全局区域的选中token
        global_selected = sum(1 for tid in selected_tokens if tid < token_info["global_tokens"])
        title = f'Global View\n{self.image_size}×{self.image_size}\nTokens: 0-{token_info["global_tokens"]-1}'
        if global_selected > 0:
            title += f'\nSelected: {global_selected} tokens'
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
        
        # 4+. 显示所有局部切片（带token网格、编号和选中标记）
        local_start_token = token_info['global_tokens'] + token_info['separator_tokens']
        
        for i, patch in enumerate(result['local_patches']):
            if 4 + i > rows * cols:
                break
            ax = plt.subplot(rows, cols, 4 + i)
            
            # 获取patch在grid中的坐标
            row, col = result['patch_coords'][i]
            
            # 为patch添加token网格和选中标记
            patch_with_grid = self.add_token_grid_to_patch(
                patch,
                patch_type='local',
                patch_row=row,
                patch_col=col,
                num_width_tiles=num_width_tiles,
                num_height_tiles=num_height_tiles,
                start_token_id=local_start_token,
                selected_tokens=selected_tokens
            )
            
            ax.imshow(patch_with_grid)
            
            # 计算这个patch的token范围和选中统计
            h, w = token_info['h'], token_info['w']
            tokens_per_row = num_width_tiles * w + 1
            patch_start = local_start_token + row * h * tokens_per_row + col * w
            patch_end = patch_start + h * w - 1
            
            # 统计该patch中选中的token数量
            patch_selected = sum(1 for tid in selected_tokens if patch_start <= tid <= patch_end)
            
            title = f'Patch ({row},{col})\n{self.image_size}×{self.image_size}\nTokens: {patch_start}-{patch_end}'
            if patch_selected > 0:
                title += f'\nSelected: {patch_selected} tokens'
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # 添加标题信息
        main_title = f'Image Processing with Selected Tokens: {num_width_tiles}×{num_height_tiles} tiles\n'
        main_title += f'Global: {token_info["global_tokens"]} | Separator: {token_info["separator_tokens"]} | Local: {token_info["local_tokens"]} | Total: {token_info["total_tokens"]} tokens'
        
        if selected_tokens:
            main_title += f'\nSelected tokens: {len(selected_tokens)} (Red background = selected token)'
        
        fig.suptitle(main_title, fontsize=12, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return result
    
    def print_token_selection_analysis(self, npz_path: str, num_width_tiles: int, num_height_tiles: int):
        """分析并打印token选择统计信息"""
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}")
            return
        
        selected_tokens = self.load_topk_data(npz_path)
        
        print("=" * 60)
        print("Token Selection Analysis")
        print("=" * 60)
        
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        global_tokens = h * (w + 1)
        separator_tokens = 1
        
        # 统计不同类型token的选择情况
        global_count = 0
        local_count = 0
        separator_count = 0
        
        for token_id in selected_tokens:
            if token_id < global_tokens:
                global_count += 1
            elif token_id == global_tokens:
                separator_count += 1
            else:
                local_count += 1
        
        print(f"Total selected tokens: {len(selected_tokens)}")
        print(f"Global tokens selected: {global_count}/{global_tokens} ({global_count/global_tokens*100:.1f}%)")
        print(f"Local tokens selected: {local_count}")
        print(f"Separator tokens selected: {separator_count}")
        
        # 显示选中的token ID（前20个）
        sorted_tokens = sorted(list(selected_tokens))
        print(f"\nFirst 20 selected token IDs: {sorted_tokens[:20]}")
        if len(sorted_tokens) > 20:
            print(f"... and {len(sorted_tokens)-20} more")


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

    # 可视化图像处理过程（只显示选中的token）
    result = visualizer.visualize_processing_with_selected_tokens(
        image_path, 
        width_tiles, 
        height_tiles,
        npz_path=npz_path,
        save_path=f"/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/4_visualization_selected_tokens_{width_tiles}x{height_tiles}.png"
    )
    
    # 打印token选择分析
    visualizer.print_token_selection_analysis(npz_path, width_tiles, height_tiles)
