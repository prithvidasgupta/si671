from tqdm import tqdm
import pandas as pd 
from complaint_binner import ComplaintBinner

def newliner(arr):
    return ", ".join(arr)

important_parts = ['air bags', 'electrical system', 'fuel system', 'power train', 'seat belts', 'suspension', 'vehicle speed control', 'visibility', 'wheels', 'engine', 'service brakes']

unknown_parts = ['air bags', 'fuel system', 'power train', 'suspension', 'vehicle speed control', 'visibility', 'wheels', 'engine', 'service brakes', 'other component']

def create_prompt1(cell):
 return  f'Which of the following car components may lead to the following complaint? Complaint: "'+ cell +'"'

def create_prompt2(cell):
 return  f'Which of the following car components may lead to the following complaint? {newliner(important_parts)}' + ' Complaint: "'+ cell +'"'

def create_prompt3(cell):
 return  f'Which of the following car components may lead to the following complaint? {newliner(unknown_parts)}' + ' Complaint: "'+ cell +'"'

df = pd.read_csv('./data/2014-2023_hyundai_cmpl.csv')

text_to_process1 = list(df['CDESCR'].str.lower().apply(create_prompt1).values)
text_to_process2 = list(df['CDESCR'].str.lower().apply(create_prompt2).values)
text_to_process3 = list(df['CDESCR'].str.lower().apply(create_prompt3).values)

cb = ComplaintBinner('google/flan-t5-large')

l1 = []
l2 = []
l3 = []

for i in tqdm(range(len(df))):
    l1.append(cb.get_sequences(text_to_process1[i], n_queries=1))
    l2.append(cb.get_sequences(text_to_process2[i], n_queries=1))
    l3.append(cb.get_sequences(text_to_process3[i], n_queries=1))
df['LABELS_1'] = l1
df['LABELS_2'] = l2
df ['LABELS_3'] = l3
df.to_csv('Hyundai-FLANT5.csv', index=False)
