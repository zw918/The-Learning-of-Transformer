import os
from loguru import logger
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

pdf_name = '手册'
current_dir = os.path.dirname(__file__)
pdf_path = os.path.abspath(os.path.join(current_dir, '../data', f"{pdf_name}.pdf"))
output_dir = os.path.abspath(os.path.join(current_dir, '../data', pdf_name))

os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, pdf_name)

try:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 初始化数据集
    ds = PymuDocDataset(pdf_bytes)
    
    # --- 关键修改点 ---
    # 尝试不传 model_type，或者改用更基础的调用
    # 如果 doc_analyze() 报错不认识 model_type，说明它现在可能默认就是 OCR
    # 我们先尝试最原始的 apply
    infer_result = ds.apply(doc_analyze) 
    
    # 根据你 ls 的结果，这里可能有 pipe_txt() 或 pipe_ocr()
    # 先尝试 pipe_txt() 看看能否跑通全流程
    pipe_res = infer_result.pipe_txt() 

    # 生成 Markdown
    md_content = pipe_res.get_markdown(os.path.basename(output_dir))
    
    with open(f"{output_filename}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.success(f"转换成功！")

except Exception as e:
    logger.exception(f"运行失败: {e}")