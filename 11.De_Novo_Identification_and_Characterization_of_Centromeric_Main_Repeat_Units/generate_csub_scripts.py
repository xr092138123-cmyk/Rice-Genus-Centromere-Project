#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math

def generate_scripts(sh_file, csub_header_info, chunk_size, output_dir):
    """
    主函数，用于生成 csub 投递脚本。
    """
    # 1. 解析 csub 表头参数
    print("--- 步骤 1/4: 解析 csub 表头参数 ---")
    header_parts = csub_header_info.split(';')
    if len(header_parts) < 3:
        print(f"错误: 参数2 (csub表头) 格式不正确。至少需要3个由';'分隔的值。", file=sys.stderr)
        print("示例: 'job_name;8;/path/to/workdir'", file=sys.stderr)
        sys.exit(1)
        
    job_name = header_parts[0]
    num_cores = header_parts[1]
    work_dir = header_parts[2]
    
    print(f"  - 作业名 (-J, -o, -e): {job_name}")
    print(f"  - 核心数 (-n): {num_cores}")
    print(f"  - 工作目录 (-cwd, cd): {work_dir}")

    # 定义 csub 表头模板
    csub_template = f"""#!/bin/bash
#CSUB -J {job_name}
#CSUB -q c01
#CSUB -o {job_name}.%J.o
#CSUB -e {job_name}.%J.e
#CSUB -n {num_cores}
#CSUB -R span[hosts=1]
#CSUB -cwd {work_dir}
cd {work_dir}

"""
    # 提示：在输出和错误文件中加入 %J (作业ID) 是一个好习惯，可以防止多次运行时互相覆盖。

    # 2. 读取需要分析的任务列表
    print("\n--- 步骤 2/4: 读取任务文件 ---")
    try:
        with open(sh_file, 'r') as f:
            # 过滤掉空行
            tasks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 任务文件 '{sh_file}' 未找到。", file=sys.stderr)
        sys.exit(1)

    if not tasks:
        print(f"警告: 任务文件 '{sh_file}' 为空，没有生成任何脚本。", file=sys.stderr)
        sys.exit(0)
    
    print(f"成功读取 {len(tasks)} 个任务。")

    # 3. 准备输出目录
    print(f"\n--- 步骤 3/4: 准备输出目录 '{output_dir}' ---")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print("输出目录已准备就绪。")
    except OSError as e:
        print(f"错误: 无法创建输出目录 '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 拆分任务并生成脚本
    print(f"\n--- 步骤 4/4: 按每份 {chunk_size} 个任务生成脚本 ---")
    
    num_chunks = math.ceil(len(tasks) / chunk_size)
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        task_chunk = tasks[start_index:end_index]
        
        # 定义输出文件名
        output_filename = f"submit_part_{i+1}.sh"
        output_filepath = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_filepath, 'w') as f_out:
                # 写入表头
                f_out.write(csub_template)
                # 写入当前批次的任务
                for task in task_chunk:
                    f_out.write(task + '\n')
            
            print(f"  - 已生成文件: {output_filepath} (包含 {len(task_chunk)} 个任务)")
        
        except IOError as e:
            print(f"错误: 写入文件 '{output_filepath}' 时发生错误: {e}", file=sys.stderr)
            # 出现IO错误时，最好停止，避免产生不完整的结果
            sys.exit(1)
            
    print(f"\n--- 完成 ---")
    print(f"成功在目录 '{output_dir}' 中生成了 {num_chunks} 个 csub 脚本。")
    print("您可以使用 'for f in $(ls *.sh); do csub $f; done' 命令来批量投递。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将一个包含多行命令的 sh 文件拆分成多个带 csub 表头的可投递脚本。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "sh_file",
        help="包含所有待执行命令的文件，每行一个任务。"
    )
    parser.add_argument(
        "csub_header_info",
        help="csub 表头参数，用分号';'分隔。\n"
             "格式: '作业名;核心数;工作目录'\n"
             "示例: 'moddotplot;8;/share/project/analysis'"
    )
    parser.add_argument(
        "chunk_size",
        type=int,
        help="每个输出脚本中包含的任务数量。"
    )
    parser.add_argument(
        "output_dir",
        help="生成的 csub 脚本的存放目录。"
    )

    args = parser.parse_args()

    generate_scripts(args.sh_file, args.csub_header_info, args.chunk_size, args.output_dir)
