import streamlit as st
from rag import process_urls,generate_answer


st.title("Real Estate Research Tool")

url1=st.sidebar.text_input("url 1")
url2=st.sidebar.text_input("url 2")
url3=st.sidebar.text_input("url 3")

placeholder=st.empty()

process_urls_button=st.sidebar.button("Process URLS")
if process_urls_button:
    urls=[url for url in (url1,url2,url3)if url!=""]
    if len(urls)==0:
        placeholder.text("you must provide atleast one url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)


query=placeholder.text_input("questions")
if query:
    try:
        answer,sources=generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("sources:")
            for source in sources.split("\n"):
                st.write(source)

    except RuntimeError as e:
        placeholder.text("you must proceess urls first")
