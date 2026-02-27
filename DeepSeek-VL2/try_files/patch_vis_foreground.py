import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

class DeepSeekVL2Visualizer:
    def __init__(self, image_path, num_width_tiles=2, num_height_tiles=2, 
                 patch_size=14, global_view_pos="head", tile_tag="2D", 
                 output_dir="deepseek_visualization", image_size=384,
                 attention_npz_path=None):  # 新增参数
        """
        Args:
            image_path: 图像路径
            num_width_tiles: 宽度方向tile数量
            num_height_tiles: 高度方向tile数量  
            patch_size: 每个patch的边长(假设是方形)
            global_view_pos: 全局视图位置 "head" 或 "tail"
            tile_tag: "2D" 或其他
            output_dir: 输出目录
            image_size: 每个图像块resize后的尺寸 (默认384)
            attention_npz_path: attention数据的npz文件路径
        """
        self.image_path = image_path
        self.num_width_tiles = num_width_tiles
        self.num_height_tiles = num_height_tiles
        self.patch_size = patch_size
        self.global_view_pos = global_view_pos
        self.tile_tag = tile_tag
        self.image_size = image_size
        self.attention_npz_path = attention_npz_path
        
        # 创建输出目录
        self.output_dir = output_dir
        self.create_output_directory()
        
        # 计算patch数量
        self.h = self.w = int(self.image_size // self.patch_size)
        
        self.load_image()
        self.calculate_token_sequence()
        
        # 加载attention数据
        self.attention_data = None
        self.topk_data = None
        if self.attention_npz_path:
            self.load_attention_data()
    
    def load_attention_data(self):
        """加载attention npz文件"""
        try:
            data = np.load(self.attention_npz_path)
            self.attention_data = data['attention']
            self.topk_data = data['topk_idx']
            print(f"成功加载attention数据:")
            print(f"  Attention shape: {self.attention_data.shape}")
            print(f"  TopK shape: {self.topk_data.shape}")
        except Exception as e:
            print(f"加载attention数据失败: {e}")
            self.attention_data = None
            self.topk_data = None
    
    def create_output_directory(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录创建: {self.output_dir}")
    
    def load_image(self):
        """加载并预处理图像"""
        self.original_image = Image.open(self.image_path).convert('RGB')
        
    def calculate_token_sequence(self):
        """计算token序列和位置映射"""
        self.token_id = 0
        self.token_map = {}  # token_id -> (tile_type, tile_idx, patch_row, patch_col)
        
        # 计算全局视图tokens
        global_tokens = self.calculate_global_tokens()

        separator_tokens = [("separator", -1, -1, -1)]

        self.token_map[self.token_id] = ("separator", -1, -1, -1)
        self.token_id += 1
        # print("token_id", self.token_id)
        # 计算局部视图tokens  
        local_tokens = self.calculate_local_tokens()
        
        # 根据global_view_pos合并
        if self.global_view_pos == "head":
            self.all_tokens = global_tokens + [("separator", -1, -1, -1)] + local_tokens
        else:
            self.all_tokens = local_tokens + [("separator", -1, -1, -1)] + global_tokens
            
    def calculate_global_tokens(self):
        """计算全局视图的token序列"""
        tokens = []
        
        if self.tile_tag == "2D":
            # 2D模式：每行patch后添加newline
            for row in range(self.h):
                for col in range(self.w):
                    tokens.append(("global", 0, row, col))
                    self.token_map[self.token_id] = ("global", 0, row, col)
                    self.token_id += 1
                # 添加换行符
                tokens.append(("newline", 0, row, -1))
                self.token_map[self.token_id] = ("newline", 0, row, -1)
                self.token_id += 1
        else:
            # 1D模式
            for patch_idx in range(self.h * self.w):
                row, col = patch_idx // self.w, patch_idx % self.w
                tokens.append(("global", 0, row, col))
                self.token_map[self.token_id] = ("global", 0, row, col)
                self.token_id += 1
                
        return tokens


    def calculate_local_tokens(self):
        """计算局部视图的token序列"""
        tokens = []
        
        if self.tile_tag == "2D":
            # 按block顺序处理，每个block独立排列
            for tile_idx in range(self.num_height_tiles * self.num_width_tiles):
                # 每个block内部的token排列
                for row in range(self.h):
                    for col in range(self.w):
                        tokens.append(("local", tile_idx, row, col))
                        self.token_map[self.token_id] = ("local", tile_idx, row, col, tile_idx)
                        self.token_id += 1
                        
                    # 每行后添加换行符
                    tokens.append(("newline", tile_idx, row, -1))
                    self.token_map[self.token_id] = ("newline", tile_idx, row, -1)
                    self.token_id += 1
        else:
            # 1D模式的局部视图处理
            for tile_idx in range(self.num_height_tiles * self.num_width_tiles):
                for patch_idx in range(self.h * self.w):
                    row, col = patch_idx // self.w, patch_idx % self.w
                    tokens.append(("local", tile_idx, row, col))
                    self.token_map[self.token_id] = ("local", tile_idx, row, col)
                    self.token_id += 1
                    
        return tokens
    # def calculate_local_tokens(self):
    #     """计算局部视图的token序列"""
    #     tokens = []
        
    #     if self.tile_tag == "2D":
    #         # 重新排列局部特征: (th*tw, h*w, d) -> (th*h, tw*w, d)
    #         for global_row in range(self.num_height_tiles * self.h):
    #             for global_col in range(self.num_width_tiles * self.w):
    #                 # 计算属于哪个tile
    #                 tile_row = global_row // self.h
    #                 tile_col = global_col // self.w
    #                 tile_idx = tile_row * self.num_width_tiles + tile_col
                    
    #                 # 计算在tile内的位置
    #                 local_row = global_row % self.h
    #                 local_col = global_col % self.w
                    
    #                 tokens.append(("local", tile_idx, local_row, local_col))
    #                 self.token_map[self.token_id] = ("local", tile_idx, local_row, local_col, 
    #                                                tile_row, tile_col, global_row, global_col)
    #                 self.token_id += 1
                    
    #             # 每行后添加换行符
    #             row_tile = global_row // self.h
    #             tokens.append(("newline", row_tile, global_row, -1))
    #             self.token_map[self.token_id] = ("newline", row_tile, global_row, -1)
    #             self.token_id += 1
    #     else:
    #         # 1D模式的局部视图处理
    #         for tile_idx in range(self.num_height_tiles * self.num_width_tiles):
    #             for patch_idx in range(self.h * self.w):
    #                 row, col = patch_idx // self.w, patch_idx % self.w
    #                 tokens.append(("local", tile_idx, row, col))
    #                 self.token_map[self.token_id] = ("local", tile_idx, row, col)
    #                 self.token_id += 1
                    
    #     return tokens
    
    def visualize_image_tiling(self, save=True):
        """可视化图像分割"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 全局视图 (整个图像的缩略版) - 修改为384x384
        global_view = self.original_image.resize((self.image_size, self.image_size))
        axes[0, 1].imshow(global_view)
        axes[0, 1].set_title(f'Global View ({self.image_size}x{self.image_size})')
        axes[0, 1].axis('off')
        
        # 绘制patch grid在全局视图上 - 使用image_size而不是固定的224
        ax = axes[0, 1]
        for i in range(self.h + 1):
            y = i * self.image_size // self.h
            ax.axhline(y=y, color='red', linewidth=1, alpha=0.7)
        for j in range(self.w + 1):
            x = j * self.image_size // self.w
            ax.axvline(x=x, color='red', linewidth=1, alpha=0.7)
            
        # 显示图像分割成tiles
        axes[0, 2].imshow(self.original_image)
        axes[0, 2].set_title(f'Image Tiling ({self.num_height_tiles}x{self.num_width_tiles})')
        
        # 绘制tile分割线
        img_width, img_height = self.original_image.size
        tile_width = img_width // self.num_width_tiles
        tile_height = img_height // self.num_height_tiles
        
        ax = axes[0, 2]
        for i in range(self.num_height_tiles + 1):
            y = i * tile_height
            ax.axhline(y=y, color='blue', linewidth=2)
        for j in range(self.num_width_tiles + 1):
            x = j * tile_width
            ax.axvline(x=x, color='blue', linewidth=2)
            
        # 标注tile编号
        for i in range(self.num_height_tiles):
            for j in range(self.num_width_tiles):
                tile_idx = i * self.num_width_tiles + j
                center_x = (j + 0.5) * tile_width
                center_y = (i + 0.5) * tile_height
                ax.text(center_x, center_y, str(tile_idx), 
                       fontsize=20, color='red', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        axes[0, 2].axis('off')
        
        # 显示局部tile示例
        if self.num_height_tiles > 0 and self.num_width_tiles > 0:
            # 裁剪第一个tile作为示例
            left = 0
            top = 0
            right = tile_width
            bottom = tile_height
            local_tile = self.original_image.crop((left, top, right, bottom))
            local_tile = local_tile.resize((self.image_size, self.image_size))  # 修改为image_size
            
            axes[1, 0].imshow(local_tile)
            axes[1, 0].set_title(f'Local Tile Example (Tile 0, {self.image_size}x{self.image_size})')
            
            # 在局部tile上绘制patch grid - 使用image_size
            ax = axes[1, 0]
            for i in range(self.h + 1):
                y = i * self.image_size // self.h
                ax.axhline(y=y, color='green', linewidth=1, alpha=0.7)
            for j in range(self.w + 1):
                x = j * self.image_size // self.w
                ax.axvline(x=x, color='green', linewidth=1, alpha=0.7)
            axes[1, 0].axis('off')
        
        # 显示token序列结构
        self.visualize_token_sequence(axes[1, 1], axes[1, 2])
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "01_image_tiling_overview.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像分割概览已保存: {save_path}")
            
        plt.show()
    
    def visualize_token_sequence(self, ax1, ax2):
        """可视化token序列"""
        # 创建token序列的可视化
        total_tokens = len(self.all_tokens)
        
        # 不同类型token的颜色
        color_map = {
            'global': 'lightblue',
            'local': 'lightgreen', 
            'newline': 'orange',
            'separator': 'red'
        }
        
        # 绘制token序列
        token_colors = []
        token_labels = []
        
        for i, (token_type, tile_idx, row, col) in enumerate(self.all_tokens[:100]):  # 只显示前100个token
            token_colors.append(color_map[token_type])
            if token_type == 'global':
                token_labels.append(f'G({row},{col})')
            elif token_type == 'local':
                token_labels.append(f'L{tile_idx}({row},{col})')
            elif token_type == 'newline':
                token_labels.append('\\n')
            else:
                token_labels.append('SEP')
        
        # 绘制token条
        y_pos = np.arange(len(token_colors))
        bars = ax1.barh(y_pos, [1]*len(token_colors), color=token_colors)
        
        ax1.set_yticks(y_pos[::5])  # 每5个显示一个标签
        ax1.set_yticklabels([token_labels[i] for i in range(0, len(token_labels), 5)])
        ax1.set_xlabel('Token Sequence (First 100 tokens)')
        ax1.set_title('Token Type Visualization')
        ax1.invert_yaxis()
        
        # 统计信息
        token_counts = {}
        for token_type, _, _, _ in self.all_tokens:
            token_counts[token_type] = token_counts.get(token_type, 0) + 1
            
        ax2.pie(token_counts.values(), labels=token_counts.keys(), autopct='%1.1f%%',
                colors=[color_map[k] for k in token_counts.keys()])
        ax2.set_title('Token Type Distribution')
    
    def create_detailed_token_map(self, save=True):
        """创建详细的token映射图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 全局视图的token映射
        global_token_map = np.zeros((self.h, self.w + 1))  # +1 for newline
        global_token_ids = []
        
        for token_id, token_info in self.token_map.items():
            if token_info[0] == 'global':
                _, _, row, col = token_info
                if col != -1:  # 不是newline
                    global_token_map[row, col] = token_id
                    global_token_ids.append(token_id)
        
        im1 = axes[0, 0].imshow(global_token_map, cmap='viridis')
        axes[0, 0].set_title('Global View Token IDs')
        axes[0, 0].set_xlabel('Patch Column (+ Newline)')
        axes[0, 0].set_ylabel('Patch Row')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 添加token ID标注
        for i in range(self.h):
            for j in range(self.w):
                if global_token_map[i, j] > 0:
                    axes[0, 0].text(j, i, str(int(global_token_map[i, j])), 
                                   ha='center', va='center', color='white', fontsize=8)
        
        # 局部视图的token映射 (合并所有tile)
        local_height = self.num_height_tiles * self.h
        local_width = self.num_width_tiles * self.w + self.num_height_tiles  # +newlines
        local_token_map = np.zeros((local_height, local_width))
        
        for token_id, token_info in self.token_map.items():
            if token_info[0] == 'local' and len(token_info) > 7:
                _, tile_idx, local_row, local_col, tile_row, tile_col, global_row, global_col = token_info
                # 考虑newline的偏移
                col_offset = global_col + global_row // self.h
                if col_offset < local_width and global_row < local_height:
                    local_token_map[global_row, col_offset] = token_id
        
        im2 = axes[0, 1].imshow(local_token_map, cmap='plasma')
        axes[0, 1].set_title('Local Views Token IDs (All Tiles Combined)')
        axes[0, 1].set_xlabel('Global Patch Column (+ Newlines)')
        axes[0, 1].set_ylabel('Global Patch Row')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 单个tile的详细视图
        if self.num_height_tiles > 0 and self.num_width_tiles > 0:
            single_tile_map = np.zeros((self.h, self.w + 1))
            target_tile = 0  # 显示第一个tile
            
            for token_id, token_info in self.token_map.items():
                if (token_info[0] == 'local' and len(token_info) > 7 and 
                    token_info[1] == target_tile):
                    _, tile_idx, local_row, local_col, tile_row, tile_col, global_row, global_col = token_info
                    single_tile_map[local_row, local_col] = token_id
            
            im3 = axes[1, 0].imshow(single_tile_map, cmap='coolwarm')
            axes[1, 0].set_title(f'Single Tile Token IDs (Tile {target_tile})')
            axes[1, 0].set_xlabel('Patch Column')
            axes[1, 0].set_ylabel('Patch Row')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # 添加token ID标注
            for i in range(self.h):
                for j in range(self.w):
                    if single_tile_map[i, j] > 0:
                        axes[1, 0].text(j, i, str(int(single_tile_map[i, j])), 
                                       ha='center', va='center', color='white', fontsize=6)
        
        # Token序列顺序图
        sequence_length = min(200, len(self.all_tokens))
        sequence_positions = np.arange(sequence_length)
        sequence_types = []
        
        type_to_num = {'global': 1, 'local': 2, 'newline': 3, 'separator': 4}
        for i in range(sequence_length):
            token_type = self.all_tokens[i][0]
            sequence_types.append(type_to_num[token_type])
        
        axes[1, 1].plot(sequence_positions, sequence_types, 'o-', markersize=3, linewidth=1)
        axes[1, 1].set_xlabel('Token Position')
        axes[1, 1].set_ylabel('Token Type')
        axes[1, 1].set_title(f'Token Sequence Order (First {sequence_length} tokens)')
        axes[1, 1].set_yticks([1, 2, 3, 4])
        axes[1, 1].set_yticklabels(['Global', 'Local', 'Newline', 'Separator'])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "02_detailed_token_mapping.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细token映射已保存: {save_path}")
        
        plt.show()
    
    def create_individual_tile_views(self, save=True):
        """创建每个tile的单独视图"""
        if self.num_height_tiles == 0 or self.num_width_tiles == 0:
            return
            
        img_width, img_height = self.original_image.size
        tile_width = img_width // self.num_width_tiles
        tile_height = img_height // self.num_height_tiles
        
        # 为每个tile创建单独的图
        for tile_row in range(self.num_height_tiles):
            for tile_col in range(self.num_width_tiles):
                tile_idx = tile_row * self.num_width_tiles + tile_col
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 裁剪tile
                left = tile_col * tile_width
                top = tile_row * tile_height
                right = (tile_col + 1) * tile_width
                bottom = (tile_row + 1) * tile_height
                
                tile_image = self.original_image.crop((left, top, right, bottom))
                tile_image_resized = tile_image.resize((self.image_size, self.image_size))  # 修改为image_size
                
                # 显示tile图像
                axes[0].imshow(tile_image_resized)
                axes[0].set_title(f'Tile {tile_idx} (Row {tile_row}, Col {tile_col})\n{self.image_size}x{self.image_size}')
                
                # 绘制patch grid - 使用image_size
                for i in range(self.h + 1):
                    y = i * self.image_size // self.h
                    axes[0].axhline(y=y, color='red', linewidth=1, alpha=0.7)
                for j in range(self.w + 1):
                    x = j * self.image_size // self.w
                    axes[0].axvline(x=x, color='red', linewidth=1, alpha=0.7)
                axes[0].axis('off')
                
                # 创建该tile的token映射
                tile_token_map = np.zeros((self.h, self.w))
                
                for token_id, token_info in self.token_map.items():
                    if (token_info[0] == 'local' and len(token_info) > 7 and 
                        token_info[1] == tile_idx):
                        _, tile_idx_info, local_row, local_col, _, _, _, _ = token_info
                        tile_token_map[local_row, local_col] = token_id
                
                im = axes[1].imshow(tile_token_map, cmap='viridis')
                axes[1].set_title(f'Token IDs for Tile {tile_idx}')
                axes[1].set_xlabel('Patch Column')
                axes[1].set_ylabel('Patch Row')
                plt.colorbar(im, ax=axes[1])
                
                # 添加token ID标注
                for i in range(self.h):
                    for j in range(self.w):
                        if tile_token_map[i, j] > 0:
                            axes[1].text(j, i, str(int(tile_token_map[i, j])), 
                                       ha='center', va='center', color='white', fontsize=8)
                
                plt.tight_layout()
                
                if save:
                    save_path = os.path.join(self.output_dir, f"03_tile_{tile_idx}_detail.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Tile {tile_idx} 详细视图已保存: {save_path}")
                
                plt.show()
    
    def create_token_sequence_visualization(self, save=True):
        """创建完整的token序列可视化"""
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        
        # 1. Token类型序列
        token_types = [token[0] for token in self.all_tokens]
        type_to_num = {'global': 1, 'local': 2, 'newline': 3, 'separator': 4}
        type_colors = {'global': 'lightblue', 'local': 'lightgreen', 
                      'newline': 'orange', 'separator': 'red'}
        
        sequence_nums = [type_to_num[t] for t in token_types]
        sequence_colors = [type_colors[t] for t in token_types]
        
        # 显示前500个token
        display_length = min(500, len(self.all_tokens))
        x_pos = np.arange(display_length)
        
        for i in range(display_length):
            axes[0].bar(i, 1, color=sequence_colors[i], width=1, alpha=0.8)
        
        axes[0].set_xlim(-0.5, display_length-0.5)
        axes[0].set_ylim(0, 1.2)
        axes[0].set_xlabel('Token Position')
        axes[0].set_title(f'Token Type Sequence (First {display_length} tokens)')
        axes[0].set_yticks([])
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=token_type.capitalize()) 
                          for token_type, color in type_colors.items()]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        # 2. Token ID序列
        if display_length > 0:
            token_ids = list(range(display_length))
            axes[1].plot(x_pos, token_ids, 'b-', linewidth=1, alpha=0.7)
            axes[1].set_xlabel('Token Position')
            axes[1].set_ylabel('Token ID')
            axes[1].set_title('Token ID Sequence')
            axes[1].grid(True, alpha=0.3)
        
        # 3. 统计信息
        token_counts = {}
        for token_type in token_types:
            token_counts[token_type] = token_counts.get(token_type, 0) + 1
        
        labels = list(token_counts.keys())
        sizes = list(token_counts.values())
        colors = [type_colors[label] for label in labels]
        
        axes[2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[2].set_title('Token Type Distribution')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "04_complete_token_sequence.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"完整token序列可视化已保存: {save_path}")
        
        plt.show()
    
    def save_statistics_report(self):
        """保存统计报告到文本文件"""
        report_path = os.path.join(self.output_dir, "05_statistics_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== DeepSeekVL2 Token Processing Statistics Report ===\n\n")
            
            # 基本信息
            f.write("Basic Configuration:\n")
            f.write(f"  Image path: {self.image_path}\n")
            f.write(f"  Image tiling: {self.num_height_tiles} x {self.num_width_tiles} = {self.num_height_tiles * self.num_width_tiles} tiles\n")
            f.write(f"  Image size per tile: {self.image_size} x {self.image_size}\n")  # 新增
            f.write(f"  Patch size: {self.patch_size} x {self.patch_size}\n")  # 新增
            f.write(f"  Patch resolution per tile: {self.h} x {self.w} = {self.h * self.w} patches\n")
            f.write(f"  Global view position: {self.global_view_pos}\n")
            f.write(f"  Tile arrangement: {self.tile_tag}\n")
            f.write(f"  Original image size: {self.original_image.size}\n\n")
            
            # Token统计
            token_counts = {}
            for token_type, _, _, _ in self.all_tokens:
                token_counts[token_type] = token_counts.get(token_type, 0) + 1
            
            total_tokens = len(self.all_tokens)
            f.write(f"Token Statistics:\n")
            f.write(f"  Total tokens: {total_tokens}\n")
            for token_type, count in token_counts.items():
                f.write(f"    {token_type.capitalize()}: {count} ({count/total_tokens*100:.1f}%)\n")
            
            f.write("\nToken Sequence Structure:\n")
            if self.global_view_pos == "head":
                f.write("  [Global patches + newlines] -> [Separator] -> [Local patches + newlines]\n")
            else:
                f.write("  [Local patches + newlines] -> [Separator] -> [Global patches + newlines]\n")
            
            # 详细token信息
            f.write(f"\nDetailed Token Information (First 50 tokens):\n")
            for i, (token_type, tile_idx, row, col) in enumerate(self.all_tokens[:50]):
                if token_type == 'global':
                    f.write(f"  Token {i:3d}: Global({row:2d},{col:2d})\n")
                elif token_type == 'local':
                    f.write(f"  Token {i:3d}: Local_Tile_{tile_idx}({row:2d},{col:2d})\n")
                elif token_type == 'newline':
                    f.write(f"  Token {i:3d}: Newline\n")
                else:
                    f.write(f"  Token {i:3d}: Separator\n")
            
            if len(self.all_tokens) > 50:
                f.write(f"  ... and {len(self.all_tokens) - 50} more tokens\n")
        
        print(f"统计报告已保存: {report_path}")
    
    def print_token_statistics(self):
        """打印token统计信息"""
        print("=== DeepSeekVL2 Token Processing Statistics ===")
        print(f"Image tiling: {self.num_height_tiles} x {self.num_width_tiles} = {self.num_height_tiles * self.num_width_tiles} tiles")
        print(f"Image size per tile: {self.image_size} x {self.image_size}")  # 新增
        print(f"Patch size: {self.patch_size} x {self.patch_size}")  # 新增
        print(f"Patch resolution per tile: {self.h} x {self.w} = {self.h * self.w} patches")
        print(f"Global view position: {self.global_view_pos}")
        print(f"Tile arrangement: {self.tile_tag}")
        print()
        
        # 统计各类token数量
        token_counts = {}
        for token_type, _, _, _ in self.all_tokens:
            token_counts[token_type] = token_counts.get(token_type, 0) + 1
        
        total_tokens = len(self.all_tokens)
        print(f"Total tokens: {total_tokens}")
        for token_type, count in token_counts.items():
            print(f"  {token_type.capitalize()}: {count} ({count/total_tokens*100:.1f}%)")
        
        print()
        print("Token sequence structure:")
        if self.global_view_pos == "head":
            print("  [Global patches + newlines] -> [Separator] -> [Local patches + newlines]")
        else:
            print("  [Local patches + newlines] -> [Separator] -> [Global patches + newlines]")

    def create_token_id_visualization_on_images(self, save=True):
        """在图像上直接可视化token ID，按序列顺序排列"""
        
        # 准备图像
        global_view = self.original_image.resize((self.image_size, self.image_size))
        
        # 准备local tiles
        local_tiles = []
        if self.num_height_tiles > 0 and self.num_width_tiles > 0:
            img_width, img_height = self.original_image.size
            tile_width = img_width // self.num_width_tiles
            tile_height = img_height // self.num_height_tiles
            
            for tile_row in range(self.num_height_tiles):
                for tile_col in range(self.num_width_tiles):
                    left = tile_col * tile_width
                    top = tile_row * tile_height
                    right = (tile_col + 1) * tile_width
                    bottom = (tile_row + 1) * tile_height
                    
                    tile_image = self.original_image.crop((left, top, right, bottom))
                    tile_image_resized = tile_image.resize((self.image_size, self.image_size))
                    local_tiles.append((tile_image_resized, tile_row * self.num_width_tiles + tile_col))
        
        # 根据global_view_pos确定排列顺序
        if self.global_view_pos == "head":
            # Global -> Separator -> Local tiles
            num_images = 1 + 1 + len(local_tiles)  # global + separator + local tiles
            image_order = ["global"] + ["separator"] + [f"local_{i}" for i in range(len(local_tiles))]
        else:
            # Local tiles -> Separator -> Global
            num_images = len(local_tiles) + 1 + 1  # local tiles + separator + global
            image_order = [f"local_{i}" for i in range(len(local_tiles))] + ["separator"] + ["global"]
        
        # 计算布局 - 进一步减少每行的图像数量
        cols = min(2, num_images)  # 每行最多2个图像，让每个图更大
        rows = (num_images + cols - 1) // cols
        
        # 大幅增加图像大小
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 15, rows * 15))  # 从10改为15
        if rows == 1:
            axes = axes.reshape(1, -1) if num_images > 1 else [axes]
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            axes[row_idx, col_idx].axis('off')
        
        current_idx = 0
        
        for img_type in image_order:
            row_idx = current_idx // cols
            col_idx = current_idx % cols
            ax = axes[row_idx, col_idx]
            
            if img_type == "global":
                self._draw_image_with_token_ids(ax, global_view, "global", 0)
                
            elif img_type == "separator":
                # 绘制separator标识
                ax.text(0.5, 0.5, "SEPARATOR", fontsize=48, ha='center', va='center',  # 进一步增加字体
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=1.0", 
                    facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=4))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title("Token Separator", fontsize=24, fontweight='bold')
                ax.axis('off')
                
            elif img_type.startswith("local_"):
                tile_idx = int(img_type.split("_")[1])
                tile_image, original_tile_idx = local_tiles[tile_idx]
                self._draw_image_with_token_ids(ax, tile_image, "local", original_tile_idx)
            
            current_idx += 1
        
        plt.tight_layout(pad=3.0)  # 增加更多padding
        
        if save:
            save_path = os.path.join(self.output_dir, "06_token_ids_on_images.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 调整dpi
            print(f"图像上的token ID可视化已保存: {save_path}")
        
        plt.show()

    def _draw_image_with_token_ids(self, ax, image, view_type, tile_idx):
        """在图像上绘制网格和token ID"""
        ax.imshow(image)
        
        # 计算patch大小和动态调整字体大小
        patch_size_pixels = self.image_size // self.h
        
        # 根据patch数量动态调整字体大小
        if self.h <= 10:
            font_size = 20
            bbox_pad = 0.4
        elif self.h <= 15:
            font_size = 16
            bbox_pad = 0.3
        elif self.h <= 20:
            font_size = 14
            bbox_pad = 0.25
        else:
            font_size = 12
            bbox_pad = 0.2
        
        print(f"Patch grid: {self.h}x{self.w}, Patch size: {patch_size_pixels}px, Font size: {font_size}")
        
        # 绘制网格 - 使用更细的线条
        for i in range(self.w + 1):
            x = i * patch_size_pixels
            ax.axvline(x=x, color='yellow', linewidth=2, alpha=0.9)
        
        for i in range(self.h + 1):
            y = i * patch_size_pixels
            ax.axhline(y=y, color='yellow', linewidth=2, alpha=0.9)
        
        # 添加token ID标注
        if view_type == "global":
            # 全局视图的token ID
            for token_id, token_info in self.token_map.items():
                if token_info[0] == "global" and len(token_info) >= 4:
                    _, _, row, col = token_info[:4]
                    if col != -1:  # 不是newline
                        center_x = (col + 0.5) * patch_size_pixels
                        center_y = (row + 0.5) * patch_size_pixels
                        ax.text(center_x, center_y, str(token_id), 
                            ha='center', va='center', color='white', 
                            fontsize=font_size, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=" + str(bbox_pad), 
                                    facecolor="blue", alpha=0.3, 
                                    edgecolor="darkblue", linewidth=2))
            
            # 添加newline标记 - 放在每一行的右侧
            for token_id, token_info in self.token_map.items():
                if token_info[0] == "newline" and len(token_info) >= 3 and token_info[1] == 0:  # global newlines
                    _, _, row, _ = token_info[:4]
                    if row >= 0 and row < self.h:  # 确保row在有效范围内
                        center_x = self.image_size + 40  # 在图像右侧
                        center_y = (row + 0.5) * patch_size_pixels  # 对应行的中心
                        ax.text(center_x, center_y, f"\\n\n({token_id})", 
                            ha='center', va='center', color='black', 
                            fontsize=font_size-2, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", 
                                    alpha=0.5, edgecolor="darkorange", linewidth=2))
            
            ax.set_title(f"Global View (Token IDs)", fontsize=22, fontweight='bold', pad=20)
            
        elif view_type == "local":
            # 局部视图的token ID
            for token_id, token_info in self.token_map.items():
                if (token_info[0] == "local" and len(token_info) > 3 and 
                    token_info[1] == tile_idx):
                    _, tile_idx_info, local_row, local_col = token_info[:4]
                    center_x = (local_col + 0.5) * patch_size_pixels
                    center_y = (local_row + 0.5) * patch_size_pixels
                    ax.text(center_x, center_y, str(token_id), 
                        ha='center', va='center', color='white', 
                        fontsize=font_size, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=" + str(bbox_pad), 
                                facecolor="green", alpha=0.3, 
                                edgecolor="darkgreen", linewidth=2))
            
            # 找到属于这个tile的newline tokens - 放在对应行的右侧
            for token_id, token_info in self.token_map.items():
                if token_info[0] == "newline" and len(token_info) >= 3:
                    # 对于local的newline，需要更仔细地处理
                    if len(token_info) > 3:  # 有更多信息的情况
                        _, tile_info, global_row, _ = token_info[:4]
                        # 计算在当前tile中的相对行位置
                        if global_row >= 0:
                            local_row = global_row % self.h
                            if local_row < self.h:
                                center_x = self.image_size + 40
                                center_y = (local_row + 0.5) * patch_size_pixels
                                ax.text(center_x, center_y, f"\\n\n({token_id})", 
                                    ha='center', va='center', color='black', 
                                    fontsize=font_size-2, fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", 
                                            alpha=0.5, edgecolor="darkorange", linewidth=2))
                    else:
                        # 简化情况：基于token在序列中的相对位置
                        _, _, row_info, _ = token_info[:4]
                        if row_info >= 0 and row_info < self.h:
                            center_x = self.image_size + 40
                            center_y = (row_info + 0.5) * patch_size_pixels
                            ax.text(center_x, center_y, f"\\n\n({token_id})", 
                                ha='center', va='center', color='black', 
                                fontsize=font_size-2, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", 
                                        alpha=0.5, edgecolor="darkorange", linewidth=2))
            
            ax.set_title(f"Local Tile {tile_idx} (Token IDs)", fontsize=22, fontweight='bold', pad=20)
        
        # 调整显示范围，在右侧留出空间显示newline
        ax.set_xlim(-30, self.image_size + 100)  # 增加右侧空间
        ax.set_ylim(self.image_size + 30, -30)   # 稍微增加上下空间
        ax.axis('off')

    def create_individual_token_visualizations(self, save=True):
        """为每个图像单独创建大尺寸的token可视化"""
        
        # 1. 创建全局视图的单独可视化
        global_view = self.original_image.resize((self.image_size, self.image_size))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        self._draw_image_with_token_ids(ax, global_view, "global", 0)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "06a_global_view_tokens.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"全局视图token可视化已保存: {save_path}")
        plt.show()
        
        # 2. 为每个local tile创建单独的可视化
        if self.num_height_tiles > 0 and self.num_width_tiles > 0:
            img_width, img_height = self.original_image.size
            tile_width = img_width // self.num_width_tiles
            tile_height = img_height // self.num_height_tiles
            
            for tile_row in range(self.num_height_tiles):
                for tile_col in range(self.num_width_tiles):
                    tile_idx = tile_row * self.num_width_tiles + tile_col
                    
                    left = tile_col * tile_width
                    top = tile_row * tile_height
                    right = (tile_col + 1) * tile_width
                    bottom = (tile_row + 1) * tile_height
                    
                    tile_image = self.original_image.crop((left, top, right, bottom))
                    tile_image_resized = tile_image.resize((self.image_size, self.image_size))
                    
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                    self._draw_image_with_token_ids(ax, tile_image_resized, "local", tile_idx)
                    plt.tight_layout()
                    
                    if save:
                        save_path = os.path.join(self.output_dir, f"06b_local_tile_{tile_idx}_tokens.png")
                        plt.savefig(save_path, dpi=200, bbox_inches='tight')
                        print(f"局部tile {tile_idx} token可视化已保存: {save_path}")
                    plt.show()


    def create_detailed_token_sequence_on_single_image(self, save=True):
        """在单个大图上显示完整的token序列"""
        
        # 计算需要的画布大小
        total_patches_width = 0
        sequence_info = []
        
        # 分析token序列，计算所需宽度
        current_row = 0
        current_col = 0
        max_width = 0
        
        for i, (token_type, tile_idx, row, col) in enumerate(self.all_tokens):
            if token_type in ["global", "local"]:
                sequence_info.append((token_type, tile_idx, current_row, current_col, i))
                current_col += 1
                max_width = max(max_width, current_col)
            elif token_type == "newline":
                sequence_info.append((token_type, tile_idx, current_row, current_col, i))
                current_row += 1
                current_col = 0
            elif token_type == "separator":
                sequence_info.append((token_type, tile_idx, current_row, current_col, i))
                current_row += 2  # separator占用更多空间
                current_col = 0
        
        # 创建大画布
        patch_size = 40  # 每个patch的显示大小
        canvas_width = max_width * patch_size + 200
        canvas_height = (current_row + 2) * patch_size + 100
        
        fig, ax = plt.subplots(1, 1, figsize=(canvas_width/50, canvas_height/50))
        
        # 准备颜色映射
        colors = {
            'global': 'lightblue',
            'local': 'lightgreen',
            'newline': 'orange',
            'separator': 'red'
        }
        
        # 绘制每个token
        for token_type, tile_idx, display_row, display_col, token_id in sequence_info:
            x = display_col * patch_size
            y = display_row * patch_size
            
            if token_type in ["global", "local"]:
                # 绘制patch方块
                rect = patches.Rectangle((x, y), patch_size-2, patch_size-2, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor=colors[token_type], alpha=0.7)
                ax.add_patch(rect)
                
                # 添加token ID
                ax.text(x + patch_size/2, y + patch_size/2, str(token_id),
                    ha='center', va='center', fontsize=10, fontweight='bold')
                
                # 添加类型标识
                type_text = "G" if token_type == "global" else f"L{tile_idx}"
                ax.text(x + patch_size/2, y + patch_size - 8, type_text,
                    ha='center', va='center', fontsize=8, style='italic')
                    
            elif token_type == "newline":
                # 绘制换行符
                ax.text(x, y + patch_size/2, f"\\n ({token_id})",
                    ha='left', va='center', fontsize=12, color='orange', fontweight='bold')
                    
            elif token_type == "separator":
                # 绘制分隔符
                ax.text(x, y + patch_size/2, f"=== SEPARATOR ({token_id}) ===",
                    ha='left', va='center', fontsize=14, color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.2))
        
        ax.set_xlim(-20, canvas_width)
        ax.set_ylim(-20, canvas_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title("Complete Token Sequence Visualization", fontsize=16, fontweight='bold')
        
        # 添加图例
        legend_elements = [patches.Patch(facecolor=color, label=token_type.capitalize()) 
                        for token_type, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "07_complete_token_sequence_single_image.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"完整token序列单图可视化已保存: {save_path}")
        
        plt.show()


    def create_topk_analysis_visualization(self, save=True):
        """创建top-k tokens的详细分析可视化"""
        
        if self.attention_data is None or self.topk_data is None:
            print("没有可用的attention数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top-K tokens的attention权重分布
        topk_weights = self.attention_data[self.topk_data]
        axes[0, 0].bar(range(len(topk_weights)), topk_weights, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Top-K Tokens Attention Weights', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Top-K Rank')
        axes[0, 0].set_ylabel('Attention Weight')
        
        # 添加token ID标注
        for i, (token_id, weight) in enumerate(zip(self.topk_data, topk_weights)):
            if i % 10 == 0:  # 每10个显示一个，避免过于拥挤
                axes[0, 0].text(i, weight, str(token_id), ha='center', va='bottom', fontsize=8)
        
        # 2. 所有tokens的attention权重直方图
        axes[0, 1].hist(self.attention_data, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(self.attention_data), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(self.attention_data):.6f}')
        axes[0, 1].axvline(np.median(self.attention_data), color='blue', linestyle='--', 
                        label=f'Median: {np.median(self.attention_data):.6f}')
        axes[0, 1].set_title('All Tokens Attention Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Token类型分析
        token_type_attention = {'global': [], 'local': [], 'other': []}
        
        for i, (token_type, _, _, _) in enumerate(self.all_tokens):
            if i < len(self.attention_data):
                if token_type == 'global':
                    token_type_attention['global'].append(self.attention_data[i])
                elif token_type == 'local':
                    token_type_attention['local'].append(self.attention_data[i])
                else:
                    token_type_attention['other'].append(self.attention_data[i])
        
        # 绘制箱线图
        data_to_plot = []
        labels = []
        for token_type, weights in token_type_attention.items():
            if weights:  # 只有当权重列表不为空时才添加
                data_to_plot.append(weights)
                labels.append(f'{token_type.capitalize()}\n(n={len(weights)})')
        
        if data_to_plot:
            axes[1, 0].boxplot(data_to_plot, labels=labels)
            axes[1, 0].set_title('Attention Weights by Token Type', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Attention Weight')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top-K tokens在序列中的位置分布
        axes[1, 1].scatter(self.topk_data, topk_weights, c=topk_weights, 
                        cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_title('Top-K Tokens Position vs Weight', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Token Position in Sequence')
        axes[1, 1].set_ylabel('Attention Weight')
        
        # 添加colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                norm=plt.Normalize(vmin=np.min(topk_weights), vmax=np.max(topk_weights)))
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1, 1], shrink=0.8)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "09_topk_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top-K分析可视化已保存: {save_path}")
        
        plt.show()


    def create_detailed_token_sequence_on_single_image_with_attention(self, save=True):
        """在图片背景上显示token序列，直观显示token与图片区域的对应关系"""
        
        if self.attention_data is None:
            return self.create_detailed_token_sequence_on_single_image(save)
        
        # 准备图片
        global_view = self.original_image.resize((self.image_size, self.image_size))
        
        # 准备local tiles
        local_tiles = []
        if self.num_height_tiles > 0 and self.num_width_tiles > 0:
            img_width, img_height = self.original_image.size
            tile_width = img_width // self.num_width_tiles
            tile_height = img_height // self.num_height_tiles
            
            for tile_row in range(self.num_height_tiles):
                for tile_col in range(self.num_width_tiles):
                    left = tile_col * tile_width
                    top = tile_row * tile_height
                    right = (tile_col + 1) * tile_width
                    bottom = (tile_row + 1) * tile_height
                    
                    tile_image = self.original_image.crop((left, top, right, bottom))
                    tile_image_resized = tile_image.resize((self.image_size, self.image_size))
                    local_tiles.append((tile_image_resized, tile_row * self.num_width_tiles + tile_col))
        
        # 归一化attention权重
        max_attention = np.max(self.attention_data)
        min_attention = np.min(self.attention_data)
        normalized_attention = (self.attention_data - min_attention) / (max_attention - min_attention + 1e-8)
        
        # 根据global_view_pos确定排列顺序
        if self.global_view_pos == "head":
            # Global -> Local tiles
            num_images = 1 + len(local_tiles)
            image_order = ["global"] + [f"local_{i}" for i in range(len(local_tiles))]
        else:
            # Local tiles -> Global
            num_images = len(local_tiles) + 1
            image_order = [f"local_{i}" for i in range(len(local_tiles))] + ["global"]
        
        # 计算布局
        cols = min(3, num_images)  # 每行最多3个图像
        rows = (num_images + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 8))
        if rows == 1:
            axes = axes.reshape(1, -1) if num_images > 1 else [axes]
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            axes[row_idx, col_idx].axis('off')
        
        current_idx = 0
        
        # 统计每个图片上的tokens
        global_tokens = {}  # {(row, col): [token_ids]}
        local_tokens = {}   # {tile_idx: {(row, col): [token_ids]}}
        
        # 收集所有tokens的位置信息
        for token_id, token_info in self.token_map.items():
            if token_info[0] == 'global' and len(token_info) >= 4:
                _, _, row, col = token_info[:4]
                if col != -1:  # 不是newline
                    if (row, col) not in global_tokens:
                        global_tokens[(row, col)] = []
                    global_tokens[(row, col)].append(token_id)
            elif token_info[0] == 'local' and len(token_info) > 3:
                _, tile_idx, local_row, local_col = token_info[:4]
                if tile_idx not in local_tokens:
                    local_tokens[tile_idx] = {}
                if (local_row, local_col) not in local_tokens[tile_idx]:
                    local_tokens[tile_idx][(local_row, local_col)] = []
                local_tokens[tile_idx][(local_row, local_col)].append(token_id)
        
        for img_type in image_order:
            row_idx = current_idx // cols
            col_idx = current_idx % cols
            ax = axes[row_idx, col_idx]
            
            if img_type == "global":
                self._draw_image_with_attention_overlay(ax, global_view, "global", 0, 
                                                    global_tokens, normalized_attention)
                
            elif img_type.startswith("local_"):
                tile_idx = int(img_type.split("_")[1])
                tile_image, original_tile_idx = local_tiles[tile_idx]
                tile_token_map = local_tokens.get(original_tile_idx, {})
                self._draw_image_with_attention_overlay(ax, tile_image, "local", original_tile_idx, 
                                                    tile_token_map, normalized_attention)
            
            current_idx += 1
        
        # 添加整体标题和统计信息
        fig.suptitle("Token Sequence on Images with Attention Weights", fontsize=16, fontweight='bold')
        
        # 添加统计信息
        topk_stats = {'global': 0, 'local': 0, 'newline': 0, 'separator': 0}
        for token_id in self.topk_data:
            if token_id < len(self.all_tokens):
                token_type = self.all_tokens[token_id][0]
                topk_stats[token_type] += 1
        
        stats_text = f"Total Tokens: {len(self.all_tokens)}\n"
        stats_text += f"Top-K Tokens: {len(self.topk_data)}\n"
        stats_text += f"Attention Range: [{min_attention:.6f}, {max_attention:.6f}]\n\n"
        stats_text += "Top-K by Type:\n"
        for token_type, count in topk_stats.items():
            if count > 0:
                stats_text += f"  {token_type}: {count}\n"
        
        # 将统计信息放在右上角
        fig.text(0.98, 0.98, stats_text, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "08_token_sequence_on_images_with_attention.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片背景的token序列可视化已保存: {save_path}")
        
        plt.show()

    def _draw_image_with_attention_overlay(self, ax, image, view_type, tile_idx, token_positions, normalized_attention):
        """在图像上绘制attention覆盖层"""
        
        # 显示图像
        ax.imshow(image, alpha=1)  # 稍微降低图像透明度，让覆盖层更明显
        
        # 计算patch大小
        patch_size_pixels = self.image_size // self.h
        
        # 绘制网格
        for i in range(self.w + 1):
            x = i * patch_size_pixels
            ax.axvline(x=x, color='white', linewidth=1, alpha=0.5)
        
        for i in range(self.h + 1):
            y = i * patch_size_pixels
            ax.axhline(y=y, color='white', linewidth=1, alpha=0.5)
        
        # 为每个patch绘制attention覆盖层
        max_tokens_in_patch = 0
        for pos, token_ids in token_positions.items():
            max_tokens_in_patch = max(max_tokens_in_patch, len(token_ids))
        
        for (row, col), token_ids in token_positions.items():
            if row >= self.h or col >= self.w:
                continue
                
            # 计算patch的像素位置
            x_start = col * patch_size_pixels
            y_start = row * patch_size_pixels
            x_end = (col + 1) * patch_size_pixels
            y_end = (row + 1) * patch_size_pixels
            
            # 为每个token ID创建覆盖层
            for i, token_id in enumerate(token_ids):
                if token_id >= len(self.attention_data):
                    continue
                    
                # 获取attention权重
                attention_weight = self.attention_data[token_id]
                normalized_weight = normalized_attention[token_id]
                is_topk = token_id in self.topk_data
                
                # 根据是否为top-k选择颜色
                if is_topk:
                    base_color = 'red'
                    alpha = 0.3 + 0.5 * normalized_weight
                else:
                    base_color = 'yellow'
                    alpha = 0.1 + 0.1 * normalized_weight
                
                # 如果一个patch有多个tokens，分区域显示
                if len(token_ids) > 1:
                    # 将patch分成子区域
                    sub_width = patch_size_pixels // len(token_ids)
                    sub_x_start = x_start + i * sub_width
                    sub_x_end = sub_x_start + sub_width
                    
                    # 绘制子区域覆盖
                    rect = patches.Rectangle((sub_x_start, y_start), 
                                        sub_width, patch_size_pixels,
                                        linewidth=2 if is_topk else 1, 
                                        edgecolor='darkred' if is_topk else 'orange',
                                        facecolor=base_color, alpha=alpha)
                    ax.add_patch(rect)
                    
                    # # 添加token ID标注
                    # text_x = sub_x_start + sub_width // 2
                    # text_y = y_start + patch_size_pixels // 3
                    # ax.text(text_x, text_y, str(token_id),
                    #     ha='center', va='center', color='white', 
                    #     fontsize=8, fontweight='bold',
                    #     bbox=dict(boxstyle="round,pad=0.2", 
                    #             facecolor='black', alpha=0.7))
                    
                    # 添加attention权重
                    if is_topk:
                        ax.text(text_x, y_start + 2*patch_size_pixels//3, f"{attention_weight:.3f}",
                            ha='center', va='center', color='white', 
                            fontsize=6, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.1", 
                                    facecolor='darkred', alpha=0.8))
                else:
                    # 单个token占据整个patch
                    rect = patches.Rectangle((x_start, y_start), 
                                        patch_size_pixels, patch_size_pixels,
                                        linewidth=3 if is_topk else 1, 
                                        edgecolor='darkred' if is_topk else 'orange',
                                        facecolor=base_color, alpha=alpha)
                    ax.add_patch(rect)
                    
                    # # 添加token ID标注
                    # text_x = x_start + patch_size_pixels // 2
                    # text_y = y_start + patch_size_pixels // 3
                    # ax.text(text_x, text_y, str(token_id),
                    #     ha='center', va='center', color='white', 
                    #     fontsize=12, fontweight='bold',
                    #     bbox=dict(boxstyle="round,pad=0.3", 
                    #             facecolor='black', alpha=0.7))
                    
                    # # 添加attention权重
                    # if is_topk:
                    #     ax.text(text_x, y_start + 2*patch_size_pixels//3, f"{attention_weight:.3f}",
                    #         ha='center', va='center', color='white', 
                    #         fontsize=8, fontweight='bold',
                    #         bbox=dict(boxstyle="round,pad=0.2", 
                    #                 facecolor='darkred', alpha=0.8))
        
        # 设置标题
        if view_type == "global":
            title = f"Global View (Token IDs with Attention)"
        else:
            title = f"Local Tile {tile_idx} (Token IDs with Attention)"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(self.image_size, 0)
        ax.axis('off')

    def create_attention_heatmap_overlay(self, save=True):
        """创建attention热力图覆盖在原图上"""
        
        if self.attention_data is None or self.topk_data is None:
            print("没有可用的attention数据")
            return
        
        # 准备图片
        global_view = self.original_image.resize((self.image_size, self.image_size))
        
        # 创建attention热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 全局视图的attention热力图
        global_attention_map = np.zeros((self.h, self.w))
        global_topk_map = np.zeros((self.h, self.w))
        
        for token_id, token_info in self.token_map.items():
            if token_info[0] == 'global' and len(token_info) >= 4:
                _, _, row, col = token_info[:4]
                if col != -1 and token_id < len(self.attention_data):
                    global_attention_map[row, col] = self.attention_data[token_id]
                    if token_id in self.topk_data:
                        global_topk_map[row, col] = 1
        
        # 显示图像
        axes[0].imshow(global_view, alpha=0.6)
        
        # 创建attention热力图覆盖层
        attention_overlay = axes[0].imshow(global_attention_map, 
                                        extent=[0, self.image_size, self.image_size, 0],
                                        cmap='hot', alpha=0.6, interpolation='bilinear')
        
        # 标记top-k tokens
        for row in range(self.h):
            for col in range(self.w):
                if global_topk_map[row, col] > 0:
                    x = (col + 0.5) * self.image_size / self.w
                    y = (row + 0.5) * self.image_size / self.h
                    axes[0].scatter(x, y, c='cyan', s=100, marker='*', 
                                edgecolors='blue', linewidth=2, alpha=0.9)
        
        axes[0].set_title('Global View - Attention Heatmap', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 添加colorbar
        plt.colorbar(attention_overlay, ax=axes[0], shrink=0.8, label='Attention Weight')
        
        # 2. 显示top-k tokens的详细信息
        axes[1].axis('off')
        
        # 创建top-k token的详细列表
        topk_info = []
        for i, token_id in enumerate(self.topk_data):
            if token_id < len(self.all_tokens):
                token_type, tile_idx, row, col = self.all_tokens[token_id][:4]
                attention_weight = self.attention_data[token_id]
                topk_info.append((i+1, token_id, token_type, tile_idx, row, col, attention_weight))
        
        # 显示top-k信息表格
        if topk_info:
            # 创建表格数据
            table_data = []
            for rank, token_id, token_type, tile_idx, row, col, weight in topk_info[:20]:  # 只显示前20个
                if token_type == 'global':
                    location = f"G({row},{col})"
                elif token_type == 'local':
                    location = f"L{tile_idx}({row},{col})"
                else:
                    location = token_type
                table_data.append([rank, token_id, location, f"{weight:.6f}"])
            
            # 创建表格
            col_labels = ['Rank', 'Token ID', 'Location', 'Attention']
            table = axes[1].table(cellText=table_data, colLabels=col_labels,
                                cellLoc='center', loc='center',
                                bbox=[0, 0.1, 1, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 设置表格样式
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 高亮top-5
            for i in range(1, min(6, len(table_data) + 1)):
                for j in range(len(col_labels)):
                    table[(i, j)].set_facecolor('#f1f1f2')
            
            axes[1].set_title('Top-K Tokens Details', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, "10_attention_heatmap_overlay.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention热力图覆盖已保存: {save_path}")
        
        plt.show()

    def run_complete_visualization_with_attention(self, query_token_ids=None):
        """运行包含attention的完整可视化流程"""
        print("开始完整的DeepSeekVL2可视化流程...")
        print(f"输出目录: {self.output_dir}")
        print("="*60)
        
        # 运行原有的可视化（除了最后的token序列）
        # self.print_token_statistics()
        # self.save_statistics_report()
        # self.visualize_image_tiling(save=True)
        # self.create_detailed_token_map(save=True)
        # self.create_individual_tile_views(save=True)
        # self.create_token_sequence_visualization(save=True)
        # self.create_token_id_visualization_on_images(save=True)
        # self.create_individual_token_visualizations(save=True)
        
        # 如果有attention数据，生成带attention的可视化
        if self.attention_data is not None:
            print("\n生成attention相关可视化...")
            
            # 生成带attention的完整token序列
            self.create_detailed_token_sequence_on_single_image_with_attention(save=True)
            
            # 生成attention热力图覆盖
            self.create_attention_heatmap_overlay(save=True)    
            
            # 生成top-k分析
            self.create_topk_analysis_visualization(save=True)
            
        else:
            # 没有attention数据，生成原始的token序列可视化
            self.create_detailed_token_sequence_on_single_image(save=True)
        
        print(f"\n所有可视化文件已保存到: {self.output_dir}")
        print("可视化完成!")


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = DeepSeekVL2Visualizer(
        image_path="/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg",  # 替换为你的图像路径
        num_width_tiles=1,
        num_height_tiles=2, 
        patch_size=14,
        global_view_pos="head",
        tile_tag="2D",
        output_dir="/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/",  # 输出目录
        image_size=196,  # 新增参数，指定图像尺寸为384x384
        attention_npz_path="/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/layer2_fastv_attention.npz"
    
    )
    
    # 运行完整的可视化流程
    visualizer.run_complete_visualization_with_attention()
