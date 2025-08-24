#!/usr/bin/env python3
"""
Comprehensive Evaluation Results Visualizer for KG_LM Project

This script generates detailed visualizations from evaluation JSON files in the eval/ directory.
It provides comprehensive analysis including performance comparisons, model configurations,
token efficiency, and statistical insights.
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import re
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_eval_data(eval_dir="eval"):
    """Load all evaluation JSON files and extract relevant metrics."""
    data = []
    
    json_files = glob.glob(os.path.join(eval_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {eval_dir}/")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            filename = os.path.basename(file_path)
            
            # Parse filename to extract metadata
            metadata = parse_filename(filename)
            
            # Extract metrics for each method (data is nested under 'hit_at_k')
            hit_at_k_data = content.get('hit_at_k', {})
            for method, metrics in hit_at_k_data.items():
                if isinstance(metrics, dict) and 'hit_at_1' in metrics:
                    record = {
                        'filename': filename,
                        'method': method,
                        'dataset': metadata.get('dataset', 'unknown'),
                        'model_type': metadata.get('model_type', 'unknown'),
                        'param_1': metadata.get('param_1', 'unknown'),
                        'param_2': metadata.get('param_2', 'unknown'),
                        'variant': metadata.get('variant', 'standard'),
                        'model_config': metadata.get('model_config', 'unknown'),
                        'hit_at_1': metrics.get('hit_at_1', 0),
                        'hit_at_3': metrics.get('hit_at_3', 0),
                        'hit_at_5': metrics.get('hit_at_5', 0),
                        'hit_at_10': metrics.get('hit_at_10', 0),
                        'avg_tokens': metrics.get('average_num_tokens', 0)  # Note: different key name
                    }
                    data.append(record)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return pd.DataFrame(data)

def parse_filename(filename):
    """Extract metadata from filename patterns."""
    metadata = {}
    
    # Remove .json extension
    name = filename.replace('.json', '')
    
    # Check for BASE files
    if name.startswith('BASE-'):
        metadata['dataset'] = name.replace('BASE-', '')
        metadata['model_type'] = 'baseline'
        metadata['model_config'] = 'baseline'
        return metadata
    
    # Check for eval files
    if name.startswith('eval-'):
        parts = name.split('-')
        # Robust dataset detection
        if 'simple' in name and 'questions' in name:
            metadata['dataset'] = 'simple-questions'
        elif 'web-qsp' in name:
            metadata['dataset'] = 'web-qsp'
        elif 'bite' in name:
            metadata['dataset'] = 'trirex-bite'
        else:
            metadata['dataset'] = 'trirex'

        # model type
        metadata['model_type'] = 'KG_LFM'
        
        # Extract L/D/variant via regex
        m = re.search(r"tri-KG-LFM-(\d+)-(\d+)(?:-([A-Za-z0-9\-]+))?", name)
        variant = 'standard'
        if m:
            metadata['param_1'] = m.group(1)
            metadata['param_2'] = m.group(2)
            if m.group(3):
                variant = m.group(3)
        else:
            numbers = [p for p in parts if p.isdigit()]
            if len(numbers) >= 2:
                metadata['param_1'] = numbers[0]
                metadata['param_2'] = numbers[1]
        metadata['variant'] = variant
        if 'param_1' in metadata and 'param_2' in metadata:
            metadata['model_config'] = f"L{metadata['param_1']}-D{metadata['param_2']}-{variant}"
        else:
            metadata['model_config'] = 'unknown'
    
    return metadata

def create_performance_heatmap(df, save_dir="plots"):
    """Create heatmap showing performance across datasets and methods."""
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    pivot_data = df.groupby(['dataset', 'method'])['hit_at_1'].mean().unstack(fill_value=0)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'Hit@1 Score'})
    
    plt.title('Performance Heatmap: Hit@1 by Dataset and Method', fontsize=16, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_metric_progression(df, save_dir="plots"):
    """Create plots showing Hit@k progression."""
    hit_cols = ['hit_at_1', 'hit_at_3', 'hit_at_5', 'hit_at_10']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hit@k Performance Across Datasets', fontsize=16, fontweight='bold')
    
    datasets = df['dataset'].unique()
    
    for i, dataset in enumerate(datasets[:4]):  # Show top 4 datasets
        ax = axes[i//2, i%2]
        
        dataset_data = df[df['dataset'] == dataset]
        
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]
            if len(method_data) > 0:
                hit_scores = [method_data[col].mean() for col in hit_cols]
                k_values = [1, 3, 5, 10]
                ax.plot(k_values, hit_scores, marker='o', linewidth=2, label=method)
        
        ax.set_title(f'{dataset}', fontweight='bold')
        ax.set_xlabel('k')
        ax.set_ylabel('Hit@k Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hitk_progression.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_analysis(df, save_dir="plots"):
    """Analyze the effect of different parameters on performance."""
    kg_lfm_data = df[df['method'] == 'KG_LFM'].copy()
    
    if len(kg_lfm_data) == 0:
        print("No KG_LFM data found for parameter analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Analysis for KG_LFM Method', fontsize=16, fontweight='bold')
    
    # Convert parameters to numeric where possible
    kg_lfm_data['param_1_num'] = pd.to_numeric(kg_lfm_data['param_1'], errors='coerce')
    kg_lfm_data['param_2_num'] = pd.to_numeric(kg_lfm_data['param_2'], errors='coerce')
    
    # Plot 1: Parameter 1 vs Hit@1
    if not kg_lfm_data['param_1_num'].isna().all():
        param1_grouped = kg_lfm_data.groupby('param_1_num')['hit_at_1'].mean()
        axes[0,0].bar(param1_grouped.index.astype(str), param1_grouped.values)
        axes[0,0].set_title('Parameter 1 vs Hit@1')
        axes[0,0].set_xlabel('Parameter 1')
        axes[0,0].set_ylabel('Average Hit@1')
    
    # Plot 2: Parameter 2 vs Hit@1
    if not kg_lfm_data['param_2_num'].isna().all():
        param2_grouped = kg_lfm_data.groupby('param_2_num')['hit_at_1'].mean()
        axes[0,1].bar(param2_grouped.index.astype(str), param2_grouped.values)
        axes[0,1].set_title('Parameter 2 vs Hit@1')
        axes[0,1].set_xlabel('Parameter 2')
        axes[0,1].set_ylabel('Average Hit@1')
    
    # Plot 3: Variant analysis
    variant_grouped = kg_lfm_data.groupby('variant')['hit_at_1'].mean().sort_values(ascending=False)
    axes[1,0].bar(range(len(variant_grouped)), variant_grouped.values)
    axes[1,0].set_xticks(range(len(variant_grouped)))
    axes[1,0].set_xticklabels(variant_grouped.index, rotation=45, ha='right')
    axes[1,0].set_title('Variant vs Hit@1')
    axes[1,0].set_ylabel('Average Hit@1')
    
    # Plot 4: Dataset-specific performance
    dataset_grouped = kg_lfm_data.groupby('dataset')['hit_at_1'].mean().sort_values(ascending=False)
    axes[1,1].bar(range(len(dataset_grouped)), dataset_grouped.values)
    axes[1,1].set_xticks(range(len(dataset_grouped)))
    axes[1,1].set_xticklabels(dataset_grouped.index, rotation=45, ha='right')
    axes[1,1].set_title('Dataset vs Hit@1')
    axes[1,1].set_ylabel('Average Hit@1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_token_efficiency_analysis(df, save_dir="plots"):
    """Create detailed token efficiency analysis."""
    # Filter out zero token entries
    df_tokens = df[df['avg_tokens'] > 0].copy()
    
    # Calculate efficiency metrics
    df_tokens['efficiency_hit1'] = df_tokens['hit_at_1'] / (df_tokens['avg_tokens'] / 1000)
    df_tokens['efficiency_hit5'] = df_tokens['hit_at_5'] / (df_tokens['avg_tokens'] / 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Token Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Token count vs Hit@1
    for method in df_tokens['method'].unique():
        method_data = df_tokens[df_tokens['method'] == method]
        axes[0,0].scatter(method_data['avg_tokens'], method_data['hit_at_1'], 
                         label=method, alpha=0.7, s=60)
    axes[0,0].set_xlabel('Average Tokens')
    axes[0,0].set_ylabel('Hit@1 Score')
    axes[0,0].set_title('Token Count vs Performance')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Efficiency comparison by method
    efficiency_by_method = df_tokens.groupby('method')['efficiency_hit1'].mean()
    axes[0,1].bar(range(len(efficiency_by_method)), efficiency_by_method.values)
    axes[0,1].set_xticks(range(len(efficiency_by_method)))
    axes[0,1].set_xticklabels(efficiency_by_method.index, rotation=45, ha='right')
    axes[0,1].set_title('Efficiency by Method (Hit@1 per 1k tokens)')
    axes[0,1].set_ylabel('Efficiency Score')
    
    # Plot 3: Efficiency comparison by dataset
    efficiency_by_dataset = df_tokens.groupby('dataset')['efficiency_hit1'].mean()
    axes[1,0].bar(range(len(efficiency_by_dataset)), efficiency_by_dataset.values)
    axes[1,0].set_xticks(range(len(efficiency_by_dataset)))
    axes[1,0].set_xticklabels(efficiency_by_dataset.index, rotation=45, ha='right')
    axes[1,0].set_title('Efficiency by Dataset')
    axes[1,0].set_ylabel('Efficiency Score')
    
    # Plot 4: Distribution of token counts
    axes[1,1].hist([df_tokens[df_tokens['method'] == method]['avg_tokens'] 
                   for method in df_tokens['method'].unique()], 
                  label=df_tokens['method'].unique(), alpha=0.7, bins=15)
    axes[1,1].set_xlabel('Average Tokens')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Token Counts')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'token_efficiency_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary(df, save_dir="plots"):
    """Create statistical summary and comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plot of Hit@1 by method
    df_hit1 = df[df['hit_at_1'] > 0]  # Filter out zeros for better visualization
    sns.boxplot(data=df_hit1, x='method', y='hit_at_1', ax=axes[0,0])
    axes[0,0].set_title('Hit@1 Distribution by Method')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Violin plot of Hit@5 by dataset
    df_hit5 = df[df['hit_at_5'] > 0]
    sns.violinplot(data=df_hit5, x='dataset', y='hit_at_5', ax=axes[0,1])
    axes[0,1].set_title('Hit@5 Distribution by Dataset')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Hit@1 vs Hit@5 scatter by method (more actionable than correlation matrix)
    df_scatter = df[(df['hit_at_1'] > 0) & (df['hit_at_5'] > 0)].copy()
    for method in df_scatter['method'].unique():
        sub = df_scatter[df_scatter['method'] == method]
        axes[1,0].scatter(sub['hit_at_1'], sub['hit_at_5'], label=method, alpha=0.7, s=50)
    axes[1,0].set_xlabel('Hit@1')
    axes[1,0].set_ylabel('Hit@5')
    axes[1,0].set_title('Hit@1 vs Hit@5 by Method')
    axes[1,0].legend()
    
    # Plot 4: Performance improvement over each baseline (separate bars)
    datasets_sorted = sorted(df['dataset'].unique())
    labels, imp_vs_orig, imp_vs_text = [], [], []
    for ds in datasets_sorted:
        kg = df[(df['dataset'] == ds) & (df['method'] == 'KG_LFM')]
        if kg.empty:
            continue
        k1 = kg['hit_at_1'].max()
        orig = df[(df['dataset'] == ds) & (df['method'] == 'original_LLM')]
        text = df[(df['dataset'] == ds) & (df['method'] == 'textualization')]
        if orig.empty and text.empty:
            continue
        labels.append(ds)
        if not orig.empty and orig['hit_at_1'].max() > 0:
            imp_vs_orig.append((k1 - orig['hit_at_1'].max()) / orig['hit_at_1'].max() * 100)
        else:
            imp_vs_orig.append(np.nan)
        if not text.empty and text['hit_at_1'].max() > 0:
            imp_vs_text.append((k1 - text['hit_at_1'].max()) / text['hit_at_1'].max() * 100)
        else:
            imp_vs_text.append(np.nan)
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        axes[1,1].bar(x - width/2, imp_vs_orig, width, label='vs original_LLM')
        axes[1,1].bar(x + width/2, imp_vs_text, width, label='vs textualization')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1,1].set_title('KG_LFM Improvement over Baselines (Hit@1, %)')
        axes[1,1].set_ylabel('Improvement (%)')
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_baseline_vs_model(df, save_dir="plots"):
    """Compare original_LLM, textualization, and best KG_LFM per dataset for Hit@1 and Hit@5 (kept separate)."""
    datasets = sorted(df['dataset'].unique())
    rows = []
    for ds in datasets:
        kg_df = df[(df['dataset'] == ds) & (df['method'] == 'KG_LFM')]
        orig_df = df[(df['dataset'] == ds) & (df['method'] == 'original_LLM')]
        text_df = df[(df['dataset'] == ds) & (df['method'] == 'textualization')]
        if kg_df.empty or (orig_df.empty and text_df.empty):
            continue
        rows.append({
            'dataset': ds,
            'orig_h1': orig_df['hit_at_1'].max() if not orig_df.empty else np.nan,
            'text_h1': text_df['hit_at_1'].max() if not text_df.empty else np.nan,
            'kg_h1': kg_df['hit_at_1'].max(),
            'orig_h5': orig_df['hit_at_5'].max() if not orig_df.empty else np.nan,
            'text_h5': text_df['hit_at_5'].max() if not text_df.empty else np.nan,
            'kg_h5': kg_df['hit_at_5'].max(),
        })
    if not rows:
        return
    res = pd.DataFrame(rows)
    x = np.arange(len(res))
    width = 0.25
    # Hit@1
    plt.figure(figsize=(13, 6))
    plt.bar(x - width, res['orig_h1'], width, label='original_LLM')
    plt.bar(x, res['text_h1'], width, label='textualization')
    plt.bar(x + width, res['kg_h1'], width, label='KG_LFM (best)')
    plt.xticks(x, res['dataset'], rotation=45, ha='right')
    plt.ylabel('Hit@1')
    plt.title('Baselines vs KG_LFM by Dataset (Hit@1)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baselines_vs_model_hit1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Hit@5
    plt.figure(figsize=(13, 6))
    plt.bar(x - width, res['orig_h5'], width, label='original_LLM')
    plt.bar(x, res['text_h5'], width, label='textualization')
    plt.bar(x + width, res['kg_h5'], width, label='KG_LFM (best)')
    plt.xticks(x, res['dataset'], rotation=45, ha='right')
    plt.ylabel('Hit@5')
    plt.title('Baselines vs KG_LFM by Dataset (Hit@5)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baselines_vs_model_hit5.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_relative_improvement(df, save_dir="plots"):
    """Relative improvement of best KG_LFM over each baseline per dataset (two bars per dataset)."""
    rows = []
    for ds in sorted(df['dataset'].unique()):
        kg_df = df[(df['dataset'] == ds) & (df['method'] == 'KG_LFM')]
        if kg_df.empty:
            continue
        k1 = kg_df['hit_at_1'].max(); k5 = kg_df['hit_at_5'].max()
        orig_df = df[(df['dataset'] == ds) & (df['method'] == 'original_LLM')]
        text_df = df[(df['dataset'] == ds) & (df['method'] == 'textualization')]
        if orig_df.empty and text_df.empty:
            continue
        row = {'dataset': ds}
        if not orig_df.empty:
            b1 = orig_df['hit_at_1'].max(); b5 = orig_df['hit_at_5'].max()
            row['imp_h1_vs_original'] = np.nan if b1 == 0 else (k1 - b1) / b1 * 100
            row['imp_h5_vs_original'] = np.nan if b5 == 0 else (k5 - b5) / b5 * 100
        else:
            row['imp_h1_vs_original'] = np.nan; row['imp_h5_vs_original'] = np.nan
        if not text_df.empty:
            b1 = text_df['hit_at_1'].max(); b5 = text_df['hit_at_5'].max()
            row['imp_h1_vs_textualization'] = np.nan if b1 == 0 else (k1 - b1) / b1 * 100
            row['imp_h5_vs_textualization'] = np.nan if b5 == 0 else (k5 - b5) / b5 * 100
        else:
            row['imp_h1_vs_textualization'] = np.nan; row['imp_h5_vs_textualization'] = np.nan
        rows.append(row)
    if not rows:
        return
    imp = pd.DataFrame(rows)
    x = np.arange(len(imp))
    width = 0.2
    # Hit@1
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, imp['imp_h1_vs_original'], width, label='vs original_LLM')
    plt.bar(x + width/2, imp['imp_h1_vs_textualization'], width, label='vs textualization')
    plt.xticks(x, imp['dataset'], rotation=45, ha='right')
    plt.ylabel('Relative Improvement (%)')
    plt.title('KG_LFM Relative Improvement over Baselines (Hit@1)')
    plt.axhline(0, color='red', linestyle='--', alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'relative_improvement_hit1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Hit@5
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, imp['imp_h5_vs_original'], width, label='vs original_LLM')
    plt.bar(x + width/2, imp['imp_h5_vs_textualization'], width, label='vs textualization')
    plt.xticks(x, imp['dataset'], rotation=45, ha='right')
    plt.ylabel('Relative Improvement (%)')
    plt.title('KG_LFM Relative Improvement over Baselines (Hit@5)')
    plt.axhline(0, color='red', linestyle='--', alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'relative_improvement_hit5.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_config_heatmaps(df, save_dir="plots"):
    """Heatmaps of Hit@1 over Layers (param_1) vs Hidden Dim (param_2) for KG_LFM per dataset."""
    sub = df[(df['method'] == 'KG_LFM') & df['param_1'].ne('unknown') & df['param_2'].ne('unknown')].copy()
    if sub.empty:
        return
    sub['L'] = pd.to_numeric(sub['param_1'], errors='coerce')
    sub['D'] = pd.to_numeric(sub['param_2'], errors='coerce')
    for ds in sorted(sub['dataset'].unique()):
        s = sub[sub['dataset'] == ds]
        if s.empty:
            continue
        pivot = s.pivot_table(index='L', columns='D', values='hit_at_1', aggfunc='max')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')
        plt.title(f'KG_LFM Sweep Heatmap (Hit@1) - {ds}')
        plt.xlabel('Hidden Dim (D)'); plt.ylabel('Layers (L)')
        plt.tight_layout()
        fname = f'config_heatmap_{ds.replace("/", "-")}.png'
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

def create_pareto_efficiency(df, save_dir="plots"):
    """Scatter of Hit@1 vs avg_tokens with Pareto frontier; markers per method."""
    dfx = df[df['avg_tokens'] > 0].copy()
    if dfx.empty:
        return
    plt.figure(figsize=(10, 7))
    for method, marker in [('textualization', 's'), ('original_LLM', '^'), ('KG_LFM', 'o')]:
        sub = dfx[dfx['method'] == method]
        if sub.empty:
            continue
        plt.scatter(sub['avg_tokens'], sub['hit_at_1'], label=method, alpha=0.7, s=70, marker=marker)
    plt.xlabel('Average Tokens'); plt.ylabel('Hit@1')
    plt.title('Performance vs Token Cost (Pareto)')
    plt.grid(True, alpha=0.3)
    pts = dfx[['avg_tokens', 'hit_at_1']].to_numpy()
    pts = pts[np.argsort(pts[:, 0])]
    frontier = []
    best = -1
    for x, y in pts:
        if y > best:
            frontier.append((x, y)); best = y
    if frontier:
        fx, fy = zip(*frontier)
        plt.plot(fx, fy, color='black', linestyle='--', label='Pareto frontier')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pareto_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_pareto_efficiency_by_dataset(df, save_dir="plots"):
    """Create separate Pareto plots (Hit@1 vs avg_tokens) for each dataset."""
    dsets = sorted(df['dataset'].dropna().unique())
    for ds in dsets:
        dfx = df[(df['dataset'] == ds) & (df['avg_tokens'] > 0)].copy()
        if dfx.empty:
            continue
        plt.figure(figsize=(10, 7))
        for method, marker in [('textualization', 's'), ('original_LLM', '^'), ('KG_LFM', 'o')]:
            sub = dfx[dfx['method'] == method]
            if sub.empty:
                continue
            plt.scatter(sub['avg_tokens'], sub['hit_at_1'], label=method, alpha=0.7, s=70, marker=marker)
        plt.xlabel('Average Tokens')
        plt.ylabel('Hit@1')
        plt.title(f'Performance vs Token Cost (Pareto) - {ds}')
        plt.grid(True, alpha=0.3)
        # Pareto frontier within dataset
        pts = dfx[['avg_tokens', 'hit_at_1']].to_numpy()
        pts = pts[np.argsort(pts[:, 0])]
        frontier = []
        best = -1
        for x, y in pts:
            if y > best:
                frontier.append((x, y)); best = y
        if frontier:
            fx, fy = zip(*frontier)
            plt.plot(fx, fy, color='black', linestyle='--', label='Pareto frontier')
        plt.legend()
        plt.tight_layout()
        fname = f"pareto_efficiency_{ds.replace('/', '-').replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

def export_leaderboards(df, save_dir="plots"):
    """Export CSV and Markdown leaderboard comparing baselines separately vs best KG_LFM."""
    rows = []
    for ds in sorted(df['dataset'].unique()):
        kg = df[(df['dataset'] == ds) & (df['method'] == 'KG_LFM')]
        if kg.empty:
            continue
        orig = df[(df['dataset'] == ds) & (df['method'] == 'original_LLM')]
        text = df[(df['dataset'] == ds) & (df['method'] == 'textualization')]
        best_idx = kg['hit_at_1'].idxmax()
        best_cfg = df.loc[best_idx, 'model_config'] if pd.notna(best_idx) else 'NA'
        rows.append({
            'dataset': ds,
            'original_LLM_hit1': orig['hit_at_1'].max() if not orig.empty else np.nan,
            'original_LLM_hit5': orig['hit_at_5'].max() if not orig.empty else np.nan,
            'textualization_hit1': text['hit_at_1'].max() if not text.empty else np.nan,
            'textualization_hit5': text['hit_at_5'].max() if not text.empty else np.nan,
            'KG_LFM_hit1': kg['hit_at_1'].max(),
            'KG_LFM_hit5': kg['hit_at_5'].max(),
            'best_kg_config': best_cfg
        })
    if not rows:
        return
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
    df_lead = pd.DataFrame(rows)
    df_lead.to_csv(out / 'leaderboard.csv', index=False)
    md = ["# Evaluation Leaderboard (Baselines separated)", "", df_lead.to_markdown(index=False)]
    (out / 'leaderboard.md').write_text("\n".join(md))

def print_detailed_summary(df):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION ANALYSIS")
    print("="*80)
    
    print(f"Total records: {len(df)}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    print(f"Methods: {', '.join(df['method'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY METHOD")
    print("="*80)
    
    method_summary = df.groupby('method').agg({
        'hit_at_1': ['mean', 'std', 'max'],
        'hit_at_5': ['mean', 'std', 'max'],
        'avg_tokens': ['mean', 'std']
    }).round(4)
    
    print(method_summary)
    
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    
    # Find best performing configurations
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        best_config = dataset_data.loc[dataset_data['hit_at_1'].idxmax()]
        print(f"{dataset:20s}: {best_config['method']} - Hit@1: {best_config['hit_at_1']:.3f}")
    
    print("\n" + "="*80)
    print("TOKEN EFFICIENCY LEADERS")
    print("="*80)
    
    df_with_tokens = df[df['avg_tokens'] > 0].copy()
    if len(df_with_tokens) > 0:
        df_with_tokens['efficiency'] = df_with_tokens['hit_at_1'] / (df_with_tokens['avg_tokens'] / 1000)
        top_efficient = df_with_tokens.nlargest(5, 'efficiency')
        
        for _, row in top_efficient.iterrows():
            print(f"{row['method']:15s} on {row['dataset']:15s}: {row['efficiency']:.4f} Hit@1 per 1k tokens")

def main():
    """Main execution function."""
    print("Loading evaluation data...")
    df = load_eval_data()
    
    if df.empty:
        print("No data found! Please check the eval/ directory.")
        return
    
    print(f"Loaded {len(df)} records from evaluation files")
    
    # Print detailed summary
    print_detailed_summary(df)
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate all visualizations
    print("\nGenerating comprehensive visualizations...")
    
    create_performance_heatmap(df)
    print("✓ Performance heatmap created")
    
    create_metric_progression(df)
    print("✓ Hit@k progression plots created")
    
    create_parameter_analysis(df)
    print("✓ Parameter analysis created")
    
    create_token_efficiency_analysis(df)
    print("✓ Token efficiency analysis created")
    
    create_statistical_summary(df)
    print("✓ Statistical summary created")
    
    create_baseline_vs_model(df)
    print("✓ Baseline vs Model comparison created")
    
    create_relative_improvement(df)
    print("✓ Relative improvement plot created")
    
    create_config_heatmaps(df)
    print("✓ Config sweep heatmaps created")
    
    create_pareto_efficiency(df)
    print("✓ Pareto efficiency plot created")
    
    create_pareto_efficiency_by_dataset(df)
    print("✓ Pareto efficiency plots per dataset created")
    
    export_leaderboards(df)
    print("✓ Leaderboard CSV and Markdown exported")
    
    print(f"\nAll visualizations saved to 'plots/' directory!")
    print("Generated files:")
    plot_files = glob.glob("plots/*.png")
    for plot_file in sorted(plot_files):
        print(f"  - {plot_file}")

if __name__ == "__main__":
    main()
