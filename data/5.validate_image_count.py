import os
import re
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# 创建日志系统
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

def count_images_in_markdown(file_path):
    """
    统计markdown文件中的图片数量
    匹配格式：![alt text](image_url)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 匹配图片链接的正则表达式
        image_pattern = r'!\[[^\]]*\]\([^\)]+\)'
        images = re.findall(image_pattern, content)
        
        return len(images)
    
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return -1
    except Exception as e:
        logger.error(f"读取文件时出错: {file_path}, 错误: {e}")
        return -1

def validate_image_counts(input_folder, output_folder):
    """
    验证处理前后每个文档的图片数量是否一致
    """
    results = []
    
    # 检查文件夹是否存在
    if not os.path.exists(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return None
    
    if not os.path.exists(output_folder):
        logger.error(f"输出文件夹不存在: {output_folder}")
        return None
    
    # 获取输入文件夹中的所有markdown文件
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.md')]
    
    logger.info(f"开始验证 {len(input_files)} 个markdown文件的图片数量")
    
    for filename in input_files:
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        # 统计原始文件的图片数量
        original_count = count_images_in_markdown(input_file_path)
        
        # 统计处理后文件的图片数量
        cleaned_count = count_images_in_markdown(output_file_path)
        
        # 判断是否一致
        is_consistent = (original_count == cleaned_count) and (original_count != -1) and (cleaned_count != -1)
        
        # 记录结果
        result = {
            '文件名': filename,
            '原始图片数量': original_count if original_count != -1 else '读取失败',
            '清理后图片数量': cleaned_count if cleaned_count != -1 else '读取失败',
            '是否一致': '✓' if is_consistent else '✗',
            '状态': '正常' if is_consistent else ('读取错误' if (original_count == -1 or cleaned_count == -1) else '数量不一致')
        }
        
        results.append(result)
        
        # 记录不一致的情况
        if not is_consistent:
            if original_count == -1 or cleaned_count == -1:
                logger.warning(f"文件读取失败: {filename}")
            else:
                logger.warning(f"图片数量不一致: {filename}, 原始: {original_count}, 清理后: {cleaned_count}")
    
    return results

def generate_report(results, output_file="image_validation_report.csv"):
    """
    生成验证报告
    """
    if not results:
        logger.error("没有验证结果可以生成报告")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存为CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"验证报告已保存到: {output_file}")
    
    # 计算总图片数量
    total_original_images = 0
    total_cleaned_images = 0
    valid_files = 0
    
    for result in results:
        original_count = result['原始图片数量']
        cleaned_count = result['清理后图片数量']
        
        # 只统计成功读取的文件
        if original_count != '读取失败' and cleaned_count != '读取失败':
            total_original_images += original_count
            total_cleaned_images += cleaned_count
            valid_files += 1
    
    # 打印统计信息
    total_files = len(results)
    consistent_files = len([r for r in results if r['是否一致'] == '✓'])
    inconsistent_files = total_files - consistent_files
    
    logger.info("\n=== 图片数量验证报告 ===")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"有效文件数: {valid_files}")
    logger.info(f"原始文件夹总图片数: {total_original_images}")
    logger.info(f"清理后文件夹总图片数: {total_cleaned_images}")
    logger.info(f"图片数量一致: {consistent_files}")
    logger.info(f"图片数量不一致: {inconsistent_files}")
    logger.info(f"一致率: {consistent_files/total_files*100:.2f}%")
    
    if total_original_images > 0:
        logger.info(f"图片保留率: {total_cleaned_images/total_original_images*100:.2f}%")
    
    # 打印详细表格
    logger.info("\n=== 详细验证结果 ===")
    logger.info(df.to_string(index=False))
    
    # 如果有不一致的文件，单独列出
    if inconsistent_files > 0:
        logger.info("\n=== 不一致的文件列表 ===")
        inconsistent_df = df[df['是否一致'] == '✗']
        logger.info(inconsistent_df.to_string(index=False))

def main():
    logger.info("开始验证图片数量!")
    
    # 输入和输出文件夹路径
    input_folder = "./data/doc"
    compare_folder = "./data/doc_cleaned"
    
    # 验证图片数量
    results = validate_image_counts(input_folder, compare_folder)
    
    if results:
        # 生成报告
        generate_report(results, "./data/image_validation_report.csv")
    else:
        logger.error("验证失败，无法生成报告")
    
    logger.info("图片数量验证完成!")

if __name__ == "__main__":
    main()