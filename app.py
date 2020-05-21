import utils
from tkinter import *
from tkinter import messagebox

tokenizer, max_length = utils.get_tokenizer_and_max_lenght()
model = utils.get_model()

window = Tk()
window.title('Weather Detection based on Tweet')
window.geometry('500x350')

def predict():
    text = textbox.get('1.0', 'end')
    prediction = utils.predict(str(text), model, tokenizer, max_length)
    messagebox.showinfo(title='PREDICTION', message=prediction)

textbox = Text(window)
button = Button(window, text='PREDICT' ,command=predict)
title = Label(window, text='Enter an example of a tweet')

title.place(x=50,y=25)
textbox.place(x=50, y=50, height=200, width=400)
button.place(x=50, y=300, width=400)

window.mainloop()
