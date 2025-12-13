#!/usr/bin/env python3
"""
TP测试结果对比脚本
用于对比不同TP配置下的输出结果，验证精度一致性
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def extract_outputs(log_file: Path) -> List[str]:
    """从日志文件中提取输出结果"""
    outputs = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 查找 "Output:" 后面的内容
            pattern = r'Output:\s*(.+?)(?=\n|$)'
            matches = re.findall(pattern, content, re.MULTILINE)
            outputs.extend(matches)
            
            # 也查找 "output_ids" 相关信息
            if not outputs:
                pattern = r'output_ids.*?\[(.*?)\]'
                matches = re.findall(pattern, content)
                if matches:
                    outputs.extend([f"Token IDs: {m}" for m in matches])
                    
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
    
    return outputs

def compare_outputs(outputs1: List[str], outputs2: List[str], tp1: int, tp2: int) -> Dict:
    """对比两个TP配置的输出"""
    comparison = {
        'tp1': tp1,
        'tp2': tp2,
        'outputs1': outputs1,
        'outputs2': outputs2,
        'match_count': 0,
        'total_count': max(len(outputs1), len(outputs2)),
        'matches': []
    }
    
    min_len = min(len(outputs1), len(outputs2))
    for i in range(min_len):
        out1 = outputs1[i].strip()
        out2 = outputs2[i].strip()
        
        # 简单字符串匹配（可以改进为更复杂的相似度计算）
        if out1 == out2:
            comparison['match_count'] += 1
            comparison['matches'].append(True)
        else:
            comparison['matches'].append(False)
    
    return comparison

def generate_report(results_dir: str = "./tp_test_results"):
    """生成TP测试对比报告"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    
    # 查找所有测试结果文件
    tp_files: Dict[int, List[Path]] = {}
    for tp in [1, 2, 4]:
        files = sorted(results_dir.glob(f"tp{tp}_*.log"), reverse=True)
        if files:
            tp_files[tp] = files
    
    if not tp_files:
        print("未找到测试结果文件")
        return
    
    print("=" * 60)
    print("GPT-OSS TP 测试结果对比报告")
    print("=" * 60)
    
    # 提取各TP的输出
    tp_outputs: Dict[int, List[str]] = {}
    for tp, files in tp_files.items():
        outputs = extract_outputs(files[0])
        tp_outputs[tp] = outputs
        print(f"\nTP={tp} 输出结果 ({len(outputs)} 条):")
        for i, out in enumerate(outputs[:5], 1):  # 只显示前5条
            print(f"  {i}. {out[:150]}...")
    
    # 对比结果
    if 1 in tp_outputs:
        base_outputs = tp_outputs[1]
        print("\n" + "=" * 60)
        print("精度一致性对比 (以TP=1为基准)")
        print("=" * 60)
        
        for tp in [2, 4]:
            if tp in tp_outputs:
                comparison = compare_outputs(base_outputs, tp_outputs[tp], 1, tp)
                
                print(f"\nTP=1 vs TP={tp}:")
                print(f"  总输出数: {comparison['total_count']}")
                print(f"  完全匹配: {comparison['match_count']}")
                print(f"  匹配率: {comparison['match_count']/comparison['total_count']*100:.1f}%")
                
                if comparison['match_count'] == comparison['total_count']:
                    print(f"  ✅ TP={tp} 与 TP=1 输出完全一致")
                elif comparison['match_count'] > 0:
                    print(f"  ⚠️  TP={tp} 与 TP=1 部分一致，需要详细检查")
                else:
                    print(f"  ❌ TP={tp} 与 TP=1 输出不一致")
                
                # 显示不匹配的项
                if not all(comparison['matches']):
                    print(f"\n  不匹配的输出:")
                    for i, match in enumerate(comparison['matches']):
                        if not match and i < len(comparison['outputs1']) and i < len(comparison['outputs2']):
                            print(f"    输出 {i+1}:")
                            print(f"      TP=1: {comparison['outputs1'][i][:100]}...")
                            print(f"      TP={tp}: {comparison['outputs2'][i][:100]}...")

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./tp_test_results"
    generate_report(results_dir)

