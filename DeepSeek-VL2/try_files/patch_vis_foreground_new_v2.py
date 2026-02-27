import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from PIL import Image, ImageOps, ImageDraw, ImageFont 
import torchvision.transforms as T 
import numpy as np 
import math 
import os 
from matplotlib import cm 
import matplotlib.colors as mcolors

class SimpleImageProcessingVisualizer: 
    """简化版DeepSeek-VL图像处理可视化（完整显示图像token+特殊token+分隔符token，并标注token id）"""

    def __init__(self, 
                image_size=392,
                patch_size=14,
                downsample_ratio=2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        
        # 图像变换（示例中未使用）
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
        # 局部视图tokens: (num_height_tiles * h) * (num_width_tiles * w + 1) （每行末尾+1个特殊token）
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

    # Token ID -> 位置映射（保留）
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
            h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
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

    # 颜色与绘制工具
    def _get_norm_cmap(self, attention_data, vmin=None, vmax=None):
        # 自动从 attention_data 推断全局 vmin/vmax
        if attention_data is not None and len(attention_data) > 0:
            if vmin is None:
                vmin = float(np.min(attention_data))
            if vmax is None:
                vmax = float(np.max(attention_data))
        else:
            if vmin is None: vmin = 0.0
            if vmax is None: vmax = 1.0
        if vmax == vmin:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('plasma')
        return norm, cmap, vmin, vmax

    def _attn_to_color(self, val, norm, cmap, default_rgba=(0,0,0,0), alpha=0.8):
        if val is None:
            return default_rgba
        rgba = list(cmap(norm(val)))
        rgba[3] = alpha
        return tuple(rgba)

    def _choose_text_color(self, rgba):
        # 根据背景色选择黑白文本色
        r, g, b, a = rgba
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'black' if luminance > 0.5 else 'white'

    # 计算 token 几何尺寸
    def _token_geometry(self):
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)  # 14
        token_px = self.image_size // h  # 28
        special_width = 50  # 特殊token列宽
        special_gap = 5     # 特殊列左内边距
        return h, w, token_px, special_width, special_gap

    # 绘制：全局 tokens 面板（含特殊列）
    def _draw_global_tokens_panel(self, ax, global_view, attention_data, norm, cmap):
        h, w, token_px, special_width, special_gap = self._token_geometry()
        # 构造扩展底图：image + special 列
        ext_w = self.image_size + special_width
        ext_h = self.image_size
        bg = Image.new('RGB', (ext_w, ext_h), (128,128,128))
        bg.paste(global_view, (0,0))
        ax.imshow(bg, extent=[0, ext_w, 0, ext_h], origin='upper', interpolation='nearest')
        ax.set_xlim(0, ext_w); ax.set_ylim(ext_h, 0)

        # 绘制每个token的矩形与id
        try:
            font = ImageFont.truetype("arial.ttf", size=8)
        except:
            font = ImageFont.load_default()

        # 在 matplotlib 上画矩形和文本（用轴坐标）
        for r in range(h):
            for c in range(w+1):
                token_id = r*(w+1) + c  # 全局token id
                # 几何位置
                if c < w:
                    x1 = c * token_px
                    y1 = r * token_px
                    x2 = x1 + token_px
                    y2 = y1 + token_px
                else:
                    x1 = w * token_px + special_gap
                    y1 = r * token_px
                    x2 = x1 + (special_width - 2*special_gap)
                    y2 = y1 + token_px

                # 注意力颜色
                attn_val = None
                if attention_data is not None and token_id < len(attention_data):
                    attn_val = float(attention_data[token_id])
                rgba = self._attn_to_color(attn_val, norm, cmap, default_rgba=(0,0,0,0), alpha=0.85)

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1, edgecolor='white', facecolor=rgba)
                ax.add_patch(rect)

                # 文本：token id 和（可选）注意力值
                txt_color = self._choose_text_color(rgba) if attn_val is not None else 'white'
                ax.text((x1+x2)/2, (y1+y2)/2, f"{token_id}",
                        color=txt_color, ha='center', va='center', fontsize=6)

        # 画网格线（图像token网格）
        eff = self.patch_size * self.downsample_ratio
        for i in range(w+1):
            x = i * token_px
            ax.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
        for j in range(h+1):
            y = j * token_px
            ax.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)

        ax.set_axis_off()

    # 绘制：局部 patch 面板（必要时包含特殊列）
    def _draw_local_patch_panel(self, ax, patch_img, attention_data, num_width_tiles, num_height_tiles, patch_row, patch_col): 
        h, w, token_px, special_width, special_gap = self._token_geometry() 
        tokens_per_row = num_width_tiles * w + 1 
        global_tokens = h * (w + 1) 
        local_base = global_tokens + 1
        show_special = (patch_col == num_width_tiles - 1)
        ext_w = self.image_size + (special_width if show_special else 0)
        ext_h = self.image_size

        bg = Image.new('RGB', (ext_w, ext_h), (128,128,128))
        bg.paste(patch_img, (0,0))
        # 关键：extent=[0, ext_w, ext_h, 0], origin='upper'，不要 set_ylim 反转
        ax.imshow(bg, extent=[0, ext_w, ext_h, 0], origin='upper', interpolation='nearest')

        for r in range(h):
            local_row_index = patch_row * h + r
            row_start = local_base + local_row_index * tokens_per_row
            patch_start = row_start + patch_col * w

            for c in range(w):
                token_id = patch_start + c
                x1 = c * token_px
                y1 = r * token_px
                x2 = x1 + token_px
                y2 = y1 + token_px

                attn_val = None
                if attention_data is not None and token_id < len(attention_data):
                    attn_val = float(attention_data[token_id])
                rgba = self._attn_to_color(attn_val, self._outer_norm, self._outer_cmap, default_rgba=(0,0,0,0), alpha=0.85)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1, edgecolor='white', facecolor=rgba)
                ax.add_patch(rect)

                txt_color = self._choose_text_color(rgba) if attn_val is not None else 'white'
                ax.text((x1+x2)/2, (y1+y2)/2, f"{token_id}",
                        color=txt_color, ha='center', va='center', fontsize=6)

            if show_special:
                special_id = local_base + local_row_index * tokens_per_row + num_width_tiles * w
                x1 = w * token_px + special_gap
                y1 = r * token_px
                x2 = x1 + (special_width - 2*special_gap)
                y2 = y1 + token_px

                attn_val = None
                if attention_data is not None and special_id < len(attention_data):
                    attn_val = float(attention_data[special_id])
                rgba = self._attn_to_color(attn_val, self._outer_norm, self._outer_cmap, default_rgba=(0,0,0,0), alpha=0.85)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1.2, edgecolor='red', facecolor=rgba)
                ax.add_patch(rect)

                txt_color = self._choose_text_color(rgba) if attn_val is not None else 'white'
                ax.text((x1+x2)/2, (y1+y2)/2, f"{special_id}",
                        color=txt_color, ha='center', va='center', fontsize=6)

        for i in range(w+1):
            x = i * token_px
            ax.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
        for j in range(h+1):
            y = j * token_px
            ax.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)

        ax.set_axis_off()



    def _draw_global_tokens_panel(self, ax, global_view, attention_data, norm, cmap): 
        h, w, token_px, special_width, special_gap = self._token_geometry() # 扩展底图：image + special 列 
        ext_w = self.image_size + special_width 
        ext_h = self.image_size 
        bg = Image.new('RGB', (ext_w, ext_h), (128,128,128)) 
        bg.paste(global_view, (0,0)) # 关键：extent=[0, ext_w, ext_h, 0], origin='upper'，不要再 set_ylim 反转 
        ax.imshow(bg, extent=[0, ext_w, ext_h, 0], origin='upper', interpolation='nearest')

        try:
            font = ImageFont.truetype("arial.ttf", size=8)
        except:
            font = ImageFont.load_default()

        # 绘制每个token的矩形与id（y 从上往下递增，直接用）
        for r in range(h):
            for c in range(w+1):
                token_id = r*(w+1) + c
                if c < w:
                    x1 = c * token_px
                    y1 = r * token_px
                    x2 = x1 + token_px
                    y2 = y1 + token_px
                else:
                    x1 = w * token_px + special_gap
                    y1 = r * token_px
                    x2 = x1 + (special_width - 2*special_gap)
                    y2 = y1 + token_px

                attn_val = None
                if attention_data is not None and token_id < len(attention_data):
                    attn_val = float(attention_data[token_id])
                rgba = self._attn_to_color(attn_val, norm, cmap, default_rgba=(0,0,0,0), alpha=0.85)

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1, edgecolor='white', facecolor=rgba)
                ax.add_patch(rect)

                txt_color = self._choose_text_color(rgba) if attn_val is not None else 'white'
                ax.text((x1+x2)/2, (y1+y2)/2, f"{token_id}",
                        color=txt_color, ha='center', va='center', fontsize=6)

        # 画网格线（图像token网格）
        for i in range(w+1):
            x = i * token_px
            ax.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
        for j in range(h+1):
            y = j * token_px
            ax.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)

        ax.set_axis_off()

    # 分隔符 token 面板
    def _draw_separator_panel(self, ax, attention_data, sep_id, vmin, vmax):
        width = 120
        height = 120
        ax.set_xlim(0, width); ax.set_ylim(height, 0)
        ax.set_axis_off()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('plasma')
        attn_val = None
        if attention_data is not None and sep_id < len(attention_data):
            attn_val = float(attention_data[sep_id])
        rgba = self._attn_to_color(attn_val, norm, cmap, default_rgba=(0,0,0,0), alpha=0.85)

        rect = patches.Rectangle((10, 10), width-20, height-20,
                                linewidth=1.5, edgecolor='orange', facecolor=rgba)
        ax.add_patch(rect)
        txt = f"SEP {sep_id}"
        txt_color = self._choose_text_color(rgba) if attn_val is not None else 'white'
        ax.text(width/2, height/2, txt, color=txt_color, ha='center', va='center', fontsize=10)

    # 主可视化
    def visualize_processing_with_attention(self, image_path: str, num_width_tiles: int, num_height_tiles: int, 
                                        npz_path: str = None, save_path: str = None):
        """
        可视化图像处理过程：
        - 原图
        - 全局 tokens（含特殊列），每格显示 token id 并用颜色映射注意力
        - 局部总览（红框）
        - 分隔符 token 独立面板
        - 每个局部 patch 面板（右侧patch包含本行特殊列）
        """
        result = self.process_image(image_path, num_width_tiles, num_height_tiles)
        token_info = result['token_info']
        h, w = token_info['h'], token_info['w']
        global_tokens = token_info['global_tokens']
        sep_id = global_tokens
        total_tokens = token_info['total_tokens']

        # 加载 attention
        attention_data = None
        topk_idx = None
        if npz_path and os.path.exists(npz_path):
            attention_data, topk_idx = self.load_attention_data(npz_path)
            print(f"Loaded attention data: len(attention)={len(attention_data)}, topk={len(topk_idx) if topk_idx is not None else 0}")

        # 若 attention 比 total_tokens 长，截断；若更短，特殊/部分token将显示为灰色
        att_for_range = None
        if attention_data is not None and len(attention_data) > 0:
            if len(attention_data) >= total_tokens:
                att_for_range = attention_data[:total_tokens]
            else:
                att_for_range = attention_data

        # 颜色归一化（全局范围）
        norm, cmap, vmin, vmax = self._get_norm_cmap(att_for_range)

        # 记录到实例临时属性，供子绘制函数使用
        self._outer_norm = norm
        self._outer_cmap = cmap

        # 布局
        num_patches = len(result['local_patches'])
        total_panels = 4 + num_patches  # 原图、全局、局部概览、分隔符 + N个patch
        cols = min(6, total_panels)
        rows = math.ceil(total_panels / cols)
        fig = plt.figure(figsize=(cols * 3.4, rows * 3.4))

        # 1. 原始图像
        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title(f'Original Image\n{result["original_size"]}', fontsize=12)
        ax1.axis('off')

        # 2. 全局 tokens 面板（含特殊列）
        ax2 = plt.subplot(rows, cols, 2)
        self._draw_global_tokens_panel(ax2, result['global_view'], attention_data, norm, cmap)
        ax2.set_title(f'Global Tokens h={h}, w={w} (with specials)\nIDs 0..{global_tokens-1}', fontsize=10)

        # 3. 局部视图（整体切分示意）
        ax3 = plt.subplot(rows, cols, 3)
        ax3.imshow(result['local_view'])
        ax3.set_title(f'Local View {result["target_size"]}\n({num_width_tiles}×{num_height_tiles} tiles)', fontsize=10)
        for pos in result['patch_positions']:
            rect = patches.Rectangle(
                (pos[0], pos[1]), 
                pos[2]-pos[0], pos[3]-pos[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax3.add_patch(rect)
        ax3.axis('off')

        # 4. 分隔符 token 面板
        ax4 = plt.subplot(rows, cols, 4)
        self._draw_separator_panel(ax4, attention_data, sep_id, vmin, vmax)
        ax4.set_title(f'Separator Token\nID {sep_id}', fontsize=10)

        # 5+. 每个局部 patch 面板（右列包含特殊列）
        for i, patch in enumerate(result['local_patches']):
            if 5 + i > rows * cols:
                break
            ax = plt.subplot(rows, cols, 5 + i)
            pr, pc = result['patch_coords'][i]
            self._draw_local_patch_panel(ax, patch, attention_data, 
                                        result['num_width_tiles'], result['num_height_tiles'],
                                        pr, pc)
            ax.set_title(f'Patch ({pr},{pc})', fontsize=10)

        plt.tight_layout()

        # 主标题
        main_title = f'Token-wise Visualization with Attention ({result["num_width_tiles"]}×{result["num_height_tiles"]} tiles)\n'
        main_title += f'Tokens: Global={token_info["global_tokens"]}, Sep=1, Local={token_info["local_tokens"]}, Total={token_info["total_tokens"]}'
        if attention_data is not None:
            main_title += f' | colormap=plasma, vmin={vmin:.4f}, vmax={vmax:.4f}'
        fig.suptitle(main_title, fontsize=12, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 清理临时属性
        if hasattr(self, '_outer_norm'):
            del self._outer_norm
        if hasattr(self, '_outer_cmap'):
            del self._outer_cmap

        return result

    def print_attention_analysis(self, npz_path: str, num_width_tiles: int, num_height_tiles: int):
        """打印attention统计信息（修复 topk 的索引关联）"""
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


if __name__ == "__main__": 
    # 创建可视化器 
    visualizer = SimpleImageProcessingVisualizer( image_size=392, patch_size=14, downsample_ratio=2 )

    # 设置路径
    # image_path = "/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg"
    image_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/3948.jpg"
    # image_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/3357.jpg"
    # image_path = '/mnt/public/usr/sunzhichao/DeepSeek-VL2/white_1200_800.png'

    npz_path = "/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/layer17_fastv_attention_vas_new_v2.npz"  # 替换为实际路径

    width_tiles = 2
    height_tiles = 2

    # 可视化图像处理过程（包含attention，包含特殊token与分隔符）
    result = visualizer.visualize_processing_with_attention(
        image_path, 
        width_tiles, 
        height_tiles,
        npz_path=npz_path,
        save_path=f"/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/17_pe_vis_tokens_with_specials_{width_tiles}x{height_tiles}.png"
    )

    # 打印attention分析
    visualizer.print_attention_analysis(npz_path, width_tiles, height_tiles)