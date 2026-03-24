
import os
import sys
import argparse
import urllib.request
import re
from pathlib import Path

# Project Gutenberg的镜像URL格式
GUTENBERG_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare N books from Project Gutenberg for LLM training"
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=60,
        help="Number of books to download (default: 60)"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting book ID (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gutenberg_60books",
        help="Output directory"
    )
    parser.add_argument(
        "--max-chunk-mb",
        type=float,
        default=50,
        help="Max chunk size in MB"
    )
    return parser.parse_args()


def download_book(book_id, output_dir):
    """
    下载单本书，使用多个可能的URL格式尝试
    """
    urls_to_try = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
    ]
    
    for url in urls_to_try:
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                
                # 保存原始文件
                raw_path = os.path.join(output_dir, "raw")
                os.makedirs(raw_path, exist_ok=True)
                
                filename = f"PG{book_id}.txt"
                filepath = os.path.join(raw_path, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True, content
        except:
            continue
    
    return False, None


def clean_gutenberg_text(text):
    """
    清理Gutenberg书籍的标准页眉页脚
    """
    # 移除标准页眉
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "The Project Gutenberg EBook of",
        "The Project Gutenberg eBook of",
        "*END*THE SMALL PRINT!"
    ]
    
    # 移除标准页脚
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's"
    ]
    
    # 找到开始位置
    start_idx = 0
    for marker in start_markers:
        if marker in text:
            start_idx = max(start_idx, text.find(marker) + len(marker))
            # 找到下一个换行
            next_newline = text.find('\n', start_idx)
            if next_newline != -1:
                start_idx = next_newline
    
    # 找到结束位置
    end_idx = len(text)
    for marker in end_markers:
        if marker in text:
            end_idx = min(end_idx, text.find(marker))
    
    # 截取内容
    if start_idx < end_idx:
        text = text[start_idx:end_idx]
    
    # 清理多余空白
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text


def is_valid_book(text, min_ascii_ratio=0.9):
    """
    检查是否为有效英文书籍
    """
    if len(text) < 1000:  # 至少1000字符
        return False
    
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) == 0:
        return False
    
    ratio = ascii_chars / len(text)
    return ratio >= min_ascii_ratio


def main():
    args = parse_args()
    
    print(f"Project Gutenberg {args.num_books} Books Downloader")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    downloaded = []
    book_id = args.start_id
    
    print(f"\n开始下载 {args.num_books} 本书...")
    
    while len(downloaded) < args.num_books:
        success, content = download_book(book_id, args.output_dir)
        
        if success:
            # 清理内容
            cleaned = clean_gutenberg_text(content)
            
            # 验证是否为有效英文书籍
            if is_valid_book(cleaned):
                # 保存清理后的版本
                processed_path = os.path.join(args.output_dir, "processed")
                os.makedirs(processed_path, exist_ok=True)
                
                filename = f"book_{len(downloaded)+1:03d}_PG{book_id}.txt"
                filepath = os.path.join(processed_path, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                downloaded.append({
                    'id': book_id,
                    'file': filename,
                    'chars': len(cleaned)
                })
                
                print(f"[{len(downloaded)}/{args.num_books}] ✓ Downloaded PG{book_id} ({len(cleaned)} chars)")
            else:
                print(f"  ✗ PG{book_id} skipped (invalid content)")
        else:
            print(f"  ✗ PG{book_id} not found")
        
        book_id += 1
        
        # 如果连续失败50次，可能已经没有更多书籍
        if book_id > args.start_id + args.num_books + 50 and len(downloaded) == 0:
            print("连续多次失败，停止下载")
            break
    
    # 合并为训练文件
    if downloaded:
        print(f"\n合并 {len(downloaded)} 本书为训练文件...")
        
        combined_text = ""
        max_bytes = args.max_chunk_mb * 1024 * 1024
        chunk_num = 1
        
        for book in downloaded:
            filepath = os.path.join(args.output_dir, "processed", book['file'])
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text_with_sep = text + "\n\n<|endoftext|>\n\n"
            text_bytes = text_with_sep.encode('utf-8')
            
            if len(combined_text.encode('utf-8')) + len(text_bytes) > max_bytes:
                # 写入当前块
                output_file = os.path.join(args.output_dir, f"training_data_{chunk_num}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"  Written: {output_file}")
                
                # 新块
                combined_text = text_with_sep
                chunk_num += 1
            else:
                combined_text += text_with_sep
        
        # 写入最后一块
        if combined_text:
            output_file = os.path.join(args.output_dir, f"training_data_{chunk_num}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            print(f"  Written: {output_file}")
        
        # 保存元数据
        metadata_file = os.path.join(args.output_dir, "metadata.txt")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"Total books: {len(downloaded)}\n")
            f.write(f"Total chunks: {chunk_num}\n")
            for book in downloaded:
                f.write(f"{book['id']}: {book['file']} ({book['chars']} chars)\n")
        
        print(f"\n✅ 完成！共下载 {len(downloaded)} 本书")
        print(f"📁 数据保存在: {args.output_dir}/")
        print(f"🚀 现在可以运行训练: python pretraining_simple.py --data_dir {args.output_dir}")
    else:
        print("❌ 没有成功下载任何书籍")


if __name__ == "__main__":
    main()