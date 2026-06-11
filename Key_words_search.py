import os
import re
import glob
from collections import Counter

# 需要安装的库：pdfplumber, jieba
# pip install pdfplumber jieba

import pdfplumber
import jieba

# ------------------------------ 配置 ---------------------------------
PDF_DIR = r"C:\Users"
STOPWORDS_CN = set([
    '的', '了', '和', '与', '或', '是', '在', '对', '及', '为', '有', '不', '上',
    '这', '那', '也', '都', '并', '而', '于', '以', '到', '中', '与', '等', '将',
    '被', '把', '从', '去', '就', '但', '只', '又', '可', '能', '会', '对', '还',
    '要', '之', '其', '来', '说', '我们', '他们', '它', '自己', '这个', '那个',
    '这些', '那些', '这样', '那样', '如何', '为什么', '什么', '怎么', '哪里',
    '方法', '研究', '分析', '基于', '利用', '提出', '一种', '不同', '问题',
    '过程', '结果', '通过', '使用', '进行', '具有', '如图', '所示', '本文'
])  # 可根据需要扩展

STOPWORDS_EN = set([
    'a', 'an', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'at', 'as', 'from', 'or', 'but',
    'the', 'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'he',
    'she', 'her', 'him', 'his', 'its', 'their', 'them', 'some', 'any', 'no',
    'not', 'so', 'such', 'can', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'has', 'have', 'had', 'do', 'does', 'did', 'very', 'just',
    'than', 'then', 'then', 'there', 'which', 'what', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'also', 'too', 'only', 'both', 'each','at','el',
    'between', 'among', 'into', 'through', 'during', 'without', 'within',
    'about', 'against', 'under', 'over', 'after', 'before', 'up', 'down',
    'out', 'off', 'above', 'below', 'again', 'further', 'then', 'once',
    'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'that', 'then', 'these', 'those', 'through', 'until', 'unto',
    'upon', 'with', 'without', 'after', 'before', 'upon', 'by', 'via'
])

# ------------------------------ 分词函数 ------------------------------
def extract_english_words(text):
    """提取英文单词（仅字母、连字符），转为小写"""
    # 匹配字母和连字符组成的单词（例如 state-of-the-art）
    words = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", text)
    return [w.lower() for w in words]

def extract_chinese_words(text):
    """提取中文句子并用 jieba 分词，返回词列表（仅保留长度>=2的词）"""
    # 保留中文字符（\u4e00-\u9fff）及常用标点，便于 jieba 处理
    chinese_text = re.sub(r"[^\u4e00-\u9fff]+", " ", text)
    words = jieba.lcut(chinese_text.strip())
    # 过滤：长度>=2且不是纯数字，且不在停用词表中
    filtered = [w for w in words if len(w) >= 2 and not w.isdigit()]
    return filtered

# ------------------------------ 主函数 ------------------------------
def analyze_pdfs(pdf_dir):
    # 获取所有 PDF 文件
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(pdf_dir, "*.PDF")))
    pdf_files = list(set(pdf_files))  # 去重
    total = len(pdf_files)
    print(f"📁 发现 {total} 个 PDF 文件（包含 .pdf/.PDF）\n")

    if total == 0:
        print("❌ 未找到任何 PDF 文件，请检查文件夹路径。")
        return

    en_counter = Counter()
    cn_counter = Counter()
    success_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        # 实时进度
        percent = idx / total * 100
        print(f"🔄 正在处理 [{idx}/{total}] {percent:.1f}% : {os.path.basename(pdf_path)}", end=" ")

        try:
            full_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            if not full_text.strip():
                print("⚠️ 无文本内容，跳过")
                continue

            # 英文词提取与计数
            en_words = extract_english_words(full_text)
            en_words_filtered = [w for w in en_words if w not in STOPWORDS_EN and len(w) > 1]
            en_counter.update(en_words_filtered)

            # 中文词提取与计数
            cn_words = extract_chinese_words(full_text)
            cn_words_filtered = [w for w in cn_words if w not in STOPWORDS_CN]
            cn_counter.update(cn_words_filtered)

            success_count += 1
            print("✅ 完成")
        except Exception as e:
            print(f"❌ 解析失败：{str(e)[:50]}")
            continue

    print("\n" + "="*60)
    print(f"📊 统计结果")
    print(f"成功解析的 PDF 数量：{success_count} / {total}")
    print("="*60)

    # 输出中文词频前30
    print("\n🇨🇳 中文关键词 TOP 30 （次数）:")
    for i, (word, freq) in enumerate(cn_counter.most_common(30), 1):
        print(f"{i:2d}. {word:10s} : {freq}")

    # 输出英文词频前30
    print("\n🇬🇧 英文关键词 TOP 30 （次数）:")
    for i, (word, freq) in enumerate(en_counter.most_common(30), 1):
        print(f"{i:2d}. {word:15s} : {freq}")

    # 可选：保存到文件
    with open("word_freq_results.txt", "w", encoding="utf-8") as f:
        f.write(f"成功解析PDF数量：{success_count}\n\n")
        f.write("中文词频 TOP 30:\n")
        for word, freq in cn_counter.most_common(30):
            f.write(f"{word}\t{freq}\n")
        f.write("\n英文词频 TOP 30:\n")
        for word, freq in en_counter.most_common(30):
            f.write(f"{word}\t{freq}\n")
    print("\n💾 结果已保存至 word_freq_results.txt")

if __name__ == "__main__":
    analyze_pdfs(PDF_DIR)