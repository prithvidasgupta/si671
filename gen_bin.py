from  tqdm import tqdm
import complaint_binner
import pandas as pd

columns = []
with open('./data/headers.txt', 'r', encoding='utf-8') as file:
    for line in file:
        columns.append(line.strip())

dataset = []
with open('./data/FLAT_CMPL.txt', 'r', encoding='utf-8') as file:
    for line in tqdm(file, total=1937674):
            splits = line.split('\t')
            temp = {}
            for idx in range(len(splits)):
                temp[columns[idx]] = splits[idx]
            dataset.append(temp)

df = pd.DataFrame(dataset)
mfr_df=df[df.MFR_NAME.isin(['Ford Motor Company', 'General Motors, LLC', 'Tesla, Inc.', 'Hyundai Motor America', 'Toyota Motor Corporation','Honda (American Honda Motor Co.)'])].copy()

cb = complaint_binner.ComplaintBinner('google/flan-t5-large')
prompt = 'Which faulty car component is the following text about? '

tqdm.pandas()

def generate_bins(text):
    x=cb.get_sequences(text, n_queries=2, prefix_prompt=prompt)
    return x
pd.set_option('display.max_colwidth', None) 
out = pd.DataFrame(
    {
    'input': df[(df['STATE']=='WV')][:10]['CDESCR'].values,
    'actual': df[(df['STATE']=='WV')][:10]['COMPDESC'].values,
    'predict': df[(df['STATE']=='WV')][:10]['CDESCR'].progress_apply(generate_bins).values
    })
out