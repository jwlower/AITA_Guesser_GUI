from tkinter import *
from joblib import dump, load
#import pickle
import pandas as pd
import sys
import time

root=Tk()
root.title("AITA? Let's find out!")
root.minsize(600,400)



def predict(model):
    txt=textBox.get("1.0","end-1c")
    if txt == "":
        p.configure(text = "No Input.",fg="red",font=("Comic Sans MS",24))
    else:
        p.configure(text = "Predicting...",fg="black",font=("Arial",12))

        cv = load('CV/cv_extended.joblib')
        NB = load("NB/NB_predictor_extended.joblib")
        MLP = load("MLP/MLP_predictor_extended.joblib")
        KNN = load("KNN/KNN_predictor_extended.joblib")
        RF = load("RF/RF_predictor_extended.joblib")
        txt = txt.replace('\n','')

        s = pd.Series([txt],name="body")
        s_test_cv = cv.transform(s)
        pred = NB.predict_proba(s_test_cv)
        if model == 1:
            pred = MLP.predict_proba(s_test_cv)
        if model == 2:
            pred = KNN.predict_proba(s_test_cv)
        if model == 3:
            pred = RF.predict_proba(s_test_cv)


        p.configure(    text = 'NTA chance = %'+str(int(pred[0][0]*100))+
                        ' YTA chance = %'+str(int(pred[0][1]*100))+
                        ' EOS chance = %'+str(int(pred[0][2]*100))+
                        ' NAH chance = %'+str(int(pred[0][3]*100))
                        ,fg="black",font=("Arial",12))




l = Label(root,text="Enter text and get a prediction.")
l.pack()

textBox=Text(root, height=20, width=50)
textBox.pack()
model = 0 #0 for NB


#Commit button
#command=lambda: retrieve_input() >>> just means do this when i press the button
buttonCommit0=Button(root, height=1, width=50, text="Naive Bayes (Pretty Accurate)",
                    command=lambda: predict(0))
buttonCommit0.pack(side=BOTTOM)

buttonCommit0=Button(root, height=1, width=50, text="MLP (More Accurate)",
                    command=lambda: predict(1))
buttonCommit0.pack(side=BOTTOM)

buttonCommit0=Button(root, height=1, width=50, text="KNN (WILD WEST)",
                    command=lambda: predict(2))
buttonCommit0.pack(side=BOTTOM)

buttonCommit0=Button(root, height=1, width=50, text="Random Forest (Almost always wrong)",
                    command=lambda: predict(3))
buttonCommit0.pack(side=BOTTOM)

p = Label(root,text="What will this be?")
p.pack()

mainloop()
