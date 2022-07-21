import sklearn
import nltk
import fitz 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from train_model import input_process

def load_model_and_Vectoriser():
    model=pickle.load(open("classifier.model","rb"))
    vectoriser=pickle.load(open("vectorizer.pickle","rb"))
    return model,vectoriser

if __name__=="__main__":
    model,vectoriser=load_model_and_Vectoriser()
    path=input("Enter path of files:")
    doc=fitz.open(path)
    content=''
    for page in range(len(doc)):
        content=content+doc[page].get_text()

    content=input_process(content)
    content=vectoriser.transform([content])
    pred=model.predict(content)
    if pred[0]==1:
        print("This is the document about AI")
    else:
        print("This is the document about WEB")