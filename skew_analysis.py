import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath='6 class csv.csv'
try:
    df = pd.read_csv(filepath)
    df.columns = ['Temperature', 'Luminosity', 'Radius', 'Absolute_Magnitude', 'StarType', 'Color', 'Spectral_Class']
except FileNotFoundError:
    print(f"Error: Target file '{filepath}' not found.")

features = ['Temperature', 'Luminosity', 'Radius']
print("📊 Generating distribution plots... Close the graphics window to exit.")

# Setup clean 1x3 horizontal layout
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(features):
    skew_val = df[col].skew()
    
    # Determine the action badge text dynamically
    rec = "StandardScaler" if abs(skew_val) <= 0.5 else "PowerTransformer Required"
    
    # Plot distribution curve
    sns.histplot(df[col], kde=True, ax=axes[i], color='#2b5c8f', edgecolor='white', alpha=0.7)
    
    # Minimalist, non-cluttered graph titles and labels
    axes[i].set_title(f"{col}\nSkew: {skew_val:.2f}", fontsize=11, fontweight='bold', pad=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Clean, professional text badge embedded inside the graph frame
    axes[i].text(0.95, 0.92, rec, transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#ccced1', alpha=0.9))

plt.tight_layout()
plt.show()
