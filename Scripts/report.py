import pandas as pd
import os
import re
import base64
from jinja2 import Template

def generate_explainable_report():
    # 读取CSV文件
    clinical_info = pd.read_csv('Outputs/clinical_info.csv')
    prediction = pd.read_csv('Outputs/prediction.csv')
    
    # 提取临床信息
    age = clinical_info['Age'].iloc[0]
    family_history = clinical_info['Family History'].iloc[0]
    smoke = clinical_info['Smoke'].iloc[0]
    alcohol = clinical_info['Alcohol'].iloc[0]
    menopause = clinical_info['Menopause'].iloc[0]
    
    # 提取预测概率
    n0_prob = prediction['Class_0'].iloc[0]
    n1_2_prob = prediction['Class_1'].iloc[0]
    n3_prob = prediction['Class_2'].iloc[0]
    if max(n0_prob, n1_2_prob, n3_prob) != n0_prob:
        invasive = "Yes"
    else:
        invasive = "No"
    
    # 获取热图文件
    heatmap_dir = 'Outputs'
    heatmap_files = [f for f in os.listdir(heatmap_dir) if f.endswith('.png')]
    
    # 从文件名中提取标签（word部分）
    heatmap_labels = []
    for file in heatmap_files:
        word = file.split('.')[0].split('_')[-1]
        heatmap_labels.append(word)
    
    # 组合文件和标签
    heatmap_data = list(zip(heatmap_files, heatmap_labels))
    
    # 生成结论文本
    predicted_class = "N1-2" if n1_2_prob == max(n0_prob, n1_2_prob, n3_prob) else "N0" if n0_prob == max(n0_prob, n1_2_prob, n3_prob) else "N3"
    conclusion = f"This is a {age} years old patient with invasive breast cancer. The family history and menopause may increase the risk of LNM. According to the heterogeneous area with irregular shape, angular margin, calcifications and internal colored blood flow signals, this patient most likely to be predicted as {predicted_class}. The patient would be suggested to have sentinel lymph node biopsy."
    
    # 读取HTML模板
    with open('Scripts/template.html', 'r') as f:
        template = Template(f.read())
    
    # 渲染HTML
    html_content = template.render(
        age=age,
        family_history=family_history,
        smoke=smoke,
        alcohol=alcohol,
        invasive=invasive,
        menopause=menopause,
        n0_prob=n0_prob,
        n1_2_prob=n1_2_prob,
        n3_prob=n3_prob,
        heatmap_data=heatmap_data,
        conclusion=conclusion
    )
    
    # 保存HTML文件
    with open('Outputs/explanable_report.html', 'w') as f:
        f.write(html_content)
    
    print("Generate Report Finish: explanable_report.html")

if __name__ == "__main__":
    generate_explainable_report()

