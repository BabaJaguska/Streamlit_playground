# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:15:14 2023

@author: mbelic
"""
import streamlit as st
import numpy as np
import pandas as pd
import requests

st.title('Sentimental robograf')
# df = pd.DataFrame({'lol':[1,1,1],
#                    'eee':[0,9,8]})

# if st.checkbox('show stuff'):
#     chart_data = df
#     st.write(df)
    
# op = st.selectbox('so this is it.',
#                   ('aha','no','yes'))
# st.write(op)

left, right = st.columns(2)
# left.write('aha!!')
# right.selectbox('ma nemoj',
#                    ('jes majke mi', 'jok'))



url_model = 'https://huggingface.co/valhalla/distilbart-mnli-12-3'

API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


input_in = left.text_area("Write something and I'll tell you how you feel", 
             'I bought a new phone but it keeps crashing!')
button  = left.button("Analyze")
payload = {
    "inputs": input_in,
    "parameters": {"candidate_labels": ["sad",
                                        "happy",
                                        "excited",
                                        "angry",
                                        "frustrated",
                                        "sarcastic",
                                        "neutral",
                                        "unsure",
                                        "disgusted"]}
    }
if button:
    output = query(payload)
    with right:
        perc = 0
        nStuffToPlot = 0
        for label, score in zip(output['labels'], output['scores']):
            st.write(f'{label}: {np.round(score,2)}')
            perc += score
            nStuffToPlot += 1
            if perc >= 0.9:
                break
        df = pd.DataFrame({'labels': output['labels'][:nStuffToPlot],
                           'scores': output['scores'][:nStuffToPlot]})
        
        st.bar_chart(df, x='labels', y='scores')
                