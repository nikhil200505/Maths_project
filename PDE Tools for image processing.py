!pip install -q numpy matplotlib scikit-image opencv-python medpy seaborn


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import data, img_as_float, filters, restoration, segmentation, util
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from medpy.filter.smoothing import anisotropic_diffusion
import cv2
import pandas as pd
import time

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

class ImageProcessor:
    def __init__(self, image):
        self.original = img_as_float(image)
        self.noisy = self._add_noise()
        self.metrics = []

    def _add_noise(self):
        np.random.seed(42)
        noisy = util.random_noise(self.original, mode='gaussian', var=0.01)
        return util.random_noise(noisy, mode='s&p', amount=0.05)

    def _edge_preservation(self, processed):
        orig_edges = filters.canny(self.original, sigma=1)
        proc_edges = filters.canny(processed, sigma=1)
        return np.mean(orig_edges == proc_edges)

    def _calculate_metrics(self, processed, method_type, is_seg=False):
        return {
            'Method': method_type[0],
            'Type': method_type[1],
            'SSIM': ssim(self.original, processed, data_range=1.0) if not is_seg else np.nan,
            'PSNR': psnr(self.original, processed, data_range=1.0) if not is_seg else np.nan,
            'Edge Preservation': self._edge_preservation(processed) if not is_seg else np.nan,
            'Dice': self._dice_coefficient(processed) if is_seg else np.nan,
            'Time (s)': processed[1],
            'Output': processed[0]
        }

    def _dice_coefficient(self, processed):
        truth = self.original > 0.5
        return 2 * np.logical_and(truth, processed).sum() / (truth.sum() + processed.sum())


processor = ImageProcessor(data.camera())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(processor.original, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processor.noisy, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.tight_layout()
plt.show()

!pip install -q scikit-image



from skimage.feature import canny
from skimage import util
from skimage import morphology


class ImageProcessor:
    def __init__(self, image):
        self.original = img_as_float(image)
        self.noisy = self._add_noise()

    def _add_noise(self):
        """Create realistic mixed noise"""
        np.random.seed(42)
        noisy = util.random_noise(self.original, mode='gaussian', var=0.01)
        return util.random_noise(noisy, mode='s&p', amount=0.05)

    def _edge_preservation(self, processed):
        """Calculate edge similarity using Canny"""
        orig_edges = canny(self.original, sigma=1)
        proc_edges = canny(processed, sigma=1)
        return np.mean(orig_edges == proc_edges)

    def _dice_coefficient(self, processed):
        """Calculate segmentation accuracy"""
        truth = self.original > 0.5
        return 2 * np.logical_and(truth, processed).sum() / (truth.sum() + processed.sum())



class TraditionalProcessor(ImageProcessor):
    def run(self):
        results = []

        # Gaussian Blur
        start = time.time()
        gaussian = filters.gaussian(self.noisy, sigma=1.5)
        results.append(self._create_result('Gaussian Blur', gaussian, start))

        # Median Filter
        start = time.time()
        median = filters.median(self.noisy, morphology.disk(3))
        results.append(self._create_result('Median Filter', median, start))

        # Bilateral Filter
        start = time.time()
        bilateral = cv2.bilateralFilter((self.noisy*255).astype(np.uint8), 9, 75, 75)/255.0
        results.append(self._create_result('Bilateral', bilateral, start))

        # Thresholding
        start = time.time()
        threshold = (self.noisy > 0.5).astype(float)
        results.append({
            'Method': 'Thresholding',
            'Type': 'Traditional',
            'SSIM': ssim(self.original, threshold, data_range=1.0),
            'PSNR': psnr(self.original, threshold, data_range=1.0),
            'Edge': self._edge_preservation(threshold),
            'Dice': self._dice_coefficient(threshold),
            'Time': time.time() - start,
            'Output': threshold
        })

        return pd.DataFrame(results)

    def _create_result(self, name, processed, start):
        return {
            'Method': name,
            'Type': 'Traditional',
            'SSIM': ssim(self.original, processed, data_range=1.0),
            'PSNR': psnr(self.original, processed, data_range=1.0),
            'Edge': self._edge_preservation(processed),
            'Dice': np.nan,
            'Time': time.time() - start,
            'Output': processed
        }


# PDE Methods Implementation
class PDEProcessor(ImageProcessor):
    def run(self):
        results = []

        # Perona-Malik
        pm = self._perona_malik()
        results.append(pm)

        # Total Variation
        tv = self._total_variation()
        results.append(tv)

        # Level Set
        levelset = self._level_set()
        results.append(levelset)

        return pd.DataFrame(results)

    def _perona_malik(self):
        start = time.time()
        processed = anisotropic_diffusion(self.noisy, niter=50, kappa=30)
        return self._create_pde_result('Perona-Malik', processed, start)

    def _total_variation(self):
        start = time.time()
        processed = restoration.denoise_tv_chambolle(self.noisy, weight=0.15)
        return self._create_pde_result('Total Variation', processed, start)

    def _level_set(self):
        start = time.time()
        processed = segmentation.chan_vese(self.noisy, mu=0.25, max_num_iter=200)
        return {
            'Method': 'Level Set',
            'Type': 'PDE',
            'SSIM': np.nan,
            'PSNR': np.nan,
            'Edge': np.nan,
            'Dice': self._dice_coefficient(processed),
            'Time': time.time() - start,
            'Output': processed
        }

    def _create_pde_result(self, name, processed, start):
        return {
            'Method': name,
            'Type': 'PDE',
            'SSIM': ssim(self.original, processed, data_range=1.0),
            'PSNR': psnr(self.original, processed, data_range=1.0),
            'Edge': self._edge_preservation(processed),
            'Dice': np.nan,
            'Time': time.time() - start,
            'Output': processed
        }



processor = ImageProcessor(data.camera())
traditional_df = TraditionalProcessor(processor.original).run()
pde_df = PDEProcessor(processor.original).run()
results = pd.concat([traditional_df, pde_df]).reset_index(drop=True)

print(results)


def create_image_grid(results):
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.2)

   
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(processor.original, cmap='gray')
    ax1.set_title("Original Image")

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(processor.noisy, cmap='gray')
    ax2.set_title("Noisy Input")


    for idx, (_, row) in enumerate(results.iterrows()):
        ax = fig.add_subplot(grid[(idx//3)+1, idx%3])
        if row['Method'] == 'Level Set':
            ax.imshow(processor.original, cmap='gray')
            ax.contour(row['Output'], colors='red', linewidths=0.8)
            ax.set_title(f"{row['Type']}: {row['Method']}\nDice: {row['Dice']:.2f}")
        else:
            ax.imshow(row['Output'], cmap='gray')
            ax.set_title(f"{row['Type']}: {row['Method']}\nSSIM: {row['SSIM']:.2f}")
        ax.axis('off')



    ax_metrics = fig.add_subplot(grid[1:, 3:])
    sns.heatmap(results[['SSIM', 'PSNR', 'Edge']].corr(),  
                annot=True, cmap='viridis', ax=ax_metrics)
    ax_metrics.set_title("Metric Correlation Matrix")

    plt.show()

create_image_grid(results)

from skimage.draw import disk
from skimage.filters import gaussian

def create_soft_blob_image(shape=(256, 256)):
    img = np.zeros(shape)
    rr1, cc1 = disk((80, 80), 40)
    rr2, cc2 = disk((160, 150), 50)
    rr3, cc3 = disk((120, 200), 30)
    img[rr1, cc1] = 1
    img[rr2, cc2] = 1
    img[rr3, cc3] = 1
    img = gaussian(img, sigma=5)
    return img_as_float(img)

class ImageProcessor:
    def __init__(self, image):
        self.original = img_as_float(image)
        self.noisy = self._add_noise()

    def _add_noise(self):
        np.random.seed(42)
        noisy = util.random_noise(self.original, mode='gaussian', var=0.01)
        return util.random_noise(noisy, mode='s&p', amount=0.05)

    def dice_score(self, processed):
        ground_truth = self.original > 0.5
        return 2 * np.logical_and(ground_truth, processed).sum() / (ground_truth.sum() + processed.sum())

def threshold_method(image):
    start = time.time()
    result = (image > 0.5).astype(float)
    return result, time.time() - start

def level_set_method(image):
    start = time.time()
    result = segmentation.chan_vese(image, mu=0.25, max_num_iter=200)
    return result, time.time() - start


image = create_soft_blob_image()
processor = ImageProcessor(image)


thresh_result, thresh_time = threshold_method(processor.noisy)
level_result, level_time = level_set_method(processor.noisy)


dice_thresh = processor.dice_score(thresh_result)
dice_level = processor.dice_score(level_result)


print(f"Dice Score - Thresholding: {dice_thresh:.4f}")
print(f"Dice Score - Level Set:    {dice_level:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(processor.original, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(thresh_result, cmap='gray')
plt.title(f'Thresholding\nDice: {dice_thresh:.2f}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(processor.original, cmap='gray')
plt.contour(level_result, colors='red')
plt.title(f'Level Set Contour\nDice: {dice_level:.2f}')
plt.axis('off')

plt.tight_layout()
plt.show()

def normalize_metrics(df):
    import pandas as pd
    df = df.copy()
    max_time = df['Time'].max()
    for col in ['SSIM', 'PSNR', 'Edge', 'Dice']:
        if col in df.columns:
            df[col] = df[col] / df[col].max()
    if 'Time' in df.columns:
        df['Time'] = 1 - (df['Time'] / max_time)
    return df

def create_radar_chart(df):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    categories = ['SSIM', 'PSNR', 'Edge', 'Dice', 'Time']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for idx, row in df.iterrows():
        try:
            values = [row[cat] if not pd.isna(row[cat]) else 0 for cat in categories]
            values += values[:1]  
            ax.plot(angles, values, linewidth=2, label=row['Method'])
            ax.fill(angles, values, alpha=0.25)
        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


normalized_results = normalize_metrics(results)
create_radar_chart(normalized_results)


def plot_metric_evolution():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))


    sns.scatterplot(data=results, x='SSIM', y='PSNR', hue='Type', style='Type', ax=axs[0,0], s=100)
    axs[0,0].set_title("SSIM vs PSNR Correlation")



    sns.barplot(data=results, x='Method', y='Edge', hue='Type', ax=axs[0,1])
    axs[0,1].set_title("Edge Preservation by Method")
    axs[0,1].tick_params(axis='x', rotation=45)



    sns.lineplot(data=results, x='Method', y='Time', hue='Type', ax=axs[1,0], marker='o')
    axs[1,0].set_title("Processing Time Comparison")
    axs[1,0].tick_params(axis='x', rotation=45)


    seg_data = results.dropna(subset=['Dice'])
    sns.violinplot(data=seg_data, x='Type', y='Dice', ax=axs[1,1])
    axs[1,1].set_title("Segmentation Accuracy Distribution")

    plt.tight_layout()
    plt.show()

plot_metric_evolution()

def parameter_sensitivity():
    weights = np.linspace(0.05, 0.3, 10)
    metrics = []

    for w in weights:
        start = time.time()
        tv = restoration.denoise_tv_chambolle(processor.noisy, weight=w)
        metrics.append({
            'Weight': w,
            'SSIM': ssim(processor.original, tv, data_range=1.0),
            'PSNR': psnr(processor.original, tv, data_range=1.0),
            'Time': time.time() - start
        })

    df = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Weight'], df['SSIM'], label='SSIM')
    ax.plot(df['Weight'], df['PSNR'], label='PSNR')
    ax.set_xlabel("TV Weight Parameter")
    ax.set_ylabel("Metric Value")
    ax2 = ax.twinx()
    ax2.plot(df['Weight'], df['Time'], color='red', linestyle='--', label='Time')
    ax2.set_ylabel("Execution Time (s)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Total Variation Parameter Sensitivity Analysis")
    plt.show()

parameter_sensitivity()
