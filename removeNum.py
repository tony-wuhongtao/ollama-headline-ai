import pandas as pd
import re

file_path = './hjb-sx.xlsx'
out_path = './hjb-sx-1.2.xlsx'
df = pd.read_excel(file_path)

def remove_num(text):
    return re.sub(r'^\d+-\d+\.\s*', '', text)

df['kpk'] = df['kp'].apply(remove_num)

df.to_excel(out_path, index=False)