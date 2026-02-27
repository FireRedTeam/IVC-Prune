import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import numpy as np
import math

class SimpleImageProcessingVisualizer:
    """简化版DeepSeek-VL图像处理可视化"""
    
    def __init__(self, 
                 image_size=196,
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
        h = w = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        
        # 全局视图tokens: h * (w + 1), +1是行分隔符
        global_tokens = h * (w + 1)
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
    
    def visualize_processing(self, image_path: str, num_width_tiles: int, num_height_tiles: int, save_path: str = None):
        """可视化图像处理过程"""
        result = self.process_image(image_path, num_width_tiles, num_height_tiles)
        
        # 计算subplot布局
        total_patches = len(result['local_patches'])
        cols = min(6, total_patches + 3)  # 3个固定图 + patches，最多6列
        rows = math.ceil((total_patches + 3) / cols)
        
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        
        # 1. 原始图像
        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title(f'Original Image\n{result["original_size"]}', fontsize=12)
        ax1.axis('off')
        
        # 2. 全局视图
        ax2 = plt.subplot(rows, cols, 2)
        ax2.imshow(result['global_view'])
        ax2.set_title(f'Global View\n{self.image_size}×{self.image_size}', fontsize=12)
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
        
        # 4+. 显示所有局部切片
        for i, patch in enumerate(result['local_patches']):
            if 4 + i > rows * cols:
                break
            ax = plt.subplot(rows, cols, 4 + i)
            ax.imshow(patch)
            row, col = result['patch_coords'][i]
            ax.set_title(f'Patch ({row},{col})\n{self.image_size}×{self.image_size}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # 添加token信息
        token_info = result['token_info']
        fig.suptitle(
            f'Image Processing: {num_width_tiles}×{num_height_tiles} tiles\n'
            f'Global: {token_info["global_tokens"]} | Local: {token_info["local_tokens"]} | Total: {token_info["total_tokens"]} tokens',
            fontsize=14, y=0.98
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return result
    
    def print_token_calculation(self, result):
        """打印token计算详情"""
        print("=" * 50)
        print("Token Calculation Details")
        print("=" * 50)
        
        token_info = result['token_info']
        h, w = token_info['h'], token_info['w']
        
        print(f"Image size: {self.image_size}×{self.image_size}")
        print(f"Patch size: {self.patch_size}")
        print(f"Downsample ratio: {self.downsample_ratio}")
        print(f"Feature map size per patch: {h}×{w}")
        print()
        
        print(f"Tiles: {result['num_width_tiles']}×{result['num_height_tiles']}")
        print(f"Total patches: {len(result['local_patches'])} (1 global + {len(result['local_patches'])} local)")
        print()
        
        print("Token breakdown:")
        print(f"  Global view: {h} × ({w} + 1) = {token_info['global_tokens']}")
        print(f"  Separator: {token_info['separator_tokens']}")
        print(f"  Local views: ({result['num_height_tiles']} × {h}) × ({result['num_width_tiles']} × {w} + 1) = {token_info['local_tokens']}")
        print(f"  Total: {token_info['total_tokens']}")


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = SimpleImageProcessingVisualizer(
        image_size=196,
        patch_size=14,
        downsample_ratio=2
    )
    
    # 创建示例图像
    def create_sample_image():
        # 创建一个带有网格模式的示例图像，便于观察切分效果
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 添加渐变背景
        for i in range(400):
            for j in range(600):
                img[i, j, 0] = int(255 * i / 400)  # 红色渐变
                img[i, j, 1] = int(255 * j / 600)  # 绿色渐变
                img[i, j, 2] = 128  # 蓝色固定
        
        # 添加网格线
        for i in range(0, 400, 50):
            img[i:i+2, :] = [255, 255, 255]  # 水平白线
        for j in range(0, 600, 50):
            img[:, j:j+2] = [255, 255, 255]  # 垂直白线
            
        return Image.fromarray(img)
    
    # 创建并保存示例图像
    sample_image = create_sample_image()
    sample_image.save("sample_grid_image.jpg")
    print("创建示例图像: sample_grid_image.jpg")
    
    image_path="/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg"
    # image = Image.open(image_path).convert('RGB')
    width_tiles = 1
    height_tiles = 2 

    result = visualizer.visualize_processing(
        image_path, 
        width_tiles, 
        height_tiles,
        f"/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/visualization_{width_tiles}x{height_tiles}.png"
    )


    # # 测试不同的tile配置
    # test_configs = [
    #     (2, 2),  # 2×2 tiles
    #     (3, 2),  # 3×2 tiles  
    #     (2, 3),  # 2×3 tiles
    # ]
    
    # for i, (width_tiles, height_tiles) in enumerate(test_configs):
    #     print(f"\n测试配置 {i+1}: {width_tiles}×{height_tiles} tiles")
    #     result = visualizer.visualize_processing(
    #         "sample_grid_image.jpg", 
    #         width_tiles, 
    #         height_tiles,
    #         f"visualization_{width_tiles}x{height_tiles}.png"
    #     )
    #     visualizer.print_token_calculation(result)
