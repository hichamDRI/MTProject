from streamlit_call import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st


st.markdown("<h1 style='text-align: center; color: red;'>Machine Translation using Transformers<br><br></h1>", unsafe_allow_html=True)

st.title('Choose your translation option :')
option = st.selectbox('',('From Frensh To Arabic', 'From Arabic To Frensh'))


text = st.text_area("Enter Text :", value='', max_chars=None, key="FrenshText")
if option=='From Frensh To Arabic':
    if st.button('Translate', key="FrenshBtn"):
        if text == '':
            st.write('Please enter Frensh text for translation') 
        else: 
            inputs = tokenizer_fr_ar.encode(text, return_tensors="pt")
            outputs = model_fr_ar.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
            out=tokenizer_fr_ar.decode(outputs[0])
            st.text_area('Translation Result :', str(out).strip('<pad>'))
    else: pass
else:
    if st.button('Translate', key="ArabicBtn"):
        if text == '':
            st.write('Please enter Arabic text for translation') 
        else: 
            inputs = tokenizer_ar_fr.encode(text, return_tensors="pt")
            outputs = model_ar_fr.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
            out=tokenizer_ar_fr.decode(outputs[0])
            st.text_area('Translation Result :', str(out).strip('<pad>'))
    else: pass


st.markdown("<h3 style='text-align: center; color: green;'>Created By: CHATER hicham & ABOURACHID Tarik</h3>", unsafe_allow_html=True)
