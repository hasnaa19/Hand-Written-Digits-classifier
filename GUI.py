import main as m
from tkinter import *

main = Tk()
main.minsize(600, 450)
main.title("Task 3")
main.config(bg='#AED6F1')


text0 = Label(main, text="                           ", bg='#AED6F1')
text0.grid(column=1, row=0)
text1 = Label(main, text="                           ", bg='#AED6F1')
text1.grid(column=1, row=1)
text2 = Label(main, text="                           ", bg='#AED6F1')
text2.grid(column=1, row=2)
text3 = Label(main, text="                           ", bg='#AED6F1')
text3.grid(column=1, row=3)


# ################################################################################################################
# ############################################# Number of hidden layers #################################################
# #################################################################################################################
def get_HL():
    HL_value = "you've entered " + HL.get() + " Hidden layers"
    Hl = Label(main, text=HL_value)
    Hl.grid(column=0, row=3)


text1 = Label(main, text="please enter Number of hidden layers: ", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=0, row=0, padx=20, pady=20)
HL = Entry(main, width=20)
HL.grid(column=0, row=1)
button = Button(main, text="submit", bg='#5DADE2',fg='#FFFFFF', command=get_HL)
button.grid(column=0, row=2)

# ################################################################################################################
# ############################################# Number of neurons hidden layers #################################################
# #################################################################################################################
def get_neurons_NO():
    neurons_NO_value = "you've entered " + neurons_NO.get() + " neurons in each hidden layers"
    neurons_No = Label(main, text=neurons_NO_value)
    neurons_No.grid(column=2, row=3)


text1 = Label(main, text="please enter Number of neurons ex:(1 2 3 4..): ", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=2, row=0, padx=20, pady=20)
neurons_NO = Entry(main, width=20)
neurons_NO.grid(column=2, row=1)
button = Button(main, text="submit", bg='#5DADE2',fg='#FFFFFF', command=get_neurons_NO)
button.grid(column=2, row=2)

################################################################################################################
############################################# choosing Dataset #################################################
#################################################################################################################
def get_DS():
    DS_value = "You chose Dataset number " + DS.get() +'.'
    DSChoice = Label(main, text=DS_value)
    DSChoice.grid(column=0, row=9)


text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=0, row=5)
text1 = Label(main, text="please choose the Dataset; 0 for penguins, and 1 for MINST:", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=0, row=6, padx=20, pady=20)
options1 = ["0", "1"]
DS = StringVar()
DS.set(options1[0])
DropDown1 = OptionMenu(main, DS, *options1)
DropDown1.grid(column=0, row=7)
button = Button(main, text="submit the choice", bg='#5DADE2',fg='#FFFFFF', command=get_DS)
button.grid(column=0, row=8)

################################################################################################################
############################################# choosing Activation function #################################################
#################################################################################################################
def get_AF():
    AF_value = AF.get()
    AFChoice = Label(main, text=AF_value)
    AFChoice.grid(column=2, row=9)


text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=2, row=5)
text1 = Label(main, text="please choose the Activation function (0: sigmoid 1:Tanh):", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=2, row=6, padx=20, pady=20)
options2 = ["0", "1"]
AF = StringVar()
AF.set(options2[0])
DropDown2 = OptionMenu(main, AF, *options2)
DropDown2.grid(column=2, row=7)
button = Button(main, text="submit the choice", bg='#5DADE2',fg='#FFFFFF', command=get_AF)
button.grid(column=2, row=8)

# ################################################################################################################
# ############################################# learning rate #################################################
# #################################################################################################################
def get_LR():
    LR_value = "you entered learning rate = " + LR.get()
    LRChoice = Label(main, text=LR_value)
    LRChoice.grid(column=0, row=14)


text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=0, row=10)
text1 = Label(main, text="please enter the learning rate :", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=0, row=11, padx=20, pady=20)
LR = Entry(main, width=20)
LR.grid(column=0, row=12)
button = Button(main, text="submit", bg='#5DADE2',fg='#FFFFFF', command=get_LR)
button.grid(column=0, row=13)


# ################################################################################################################
# ############################################# epochs number #################################################
# #################################################################################################################
def get_epochs_Num():
    EN_value = "you entered epochs number = " + EN.get()
    ENChoice = Label(main, text=EN_value)
    ENChoice.grid(column=1, row=14)


text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=1, row=10)
text1 = Label(main, text="please enter the epochs number :", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=1, row=11,padx=20, pady=20)
EN = Entry(main, width=20)
EN.grid(column=1, row=12)
button = Button(main, text="submit", bg='#5DADE2',fg='#FFFFFF', command=get_epochs_Num)
button.grid(column=1, row=13)


# ################################################################################################################
# ############################################# biased or not #################################################
# #################################################################################################################
def check_bias():
    CBchoice = Label(main, text=var.get())
    CBchoice.grid(column=2, row=14)


text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=2, row=10)
text1 = Label(main, text="Adding bias:", bg='#2E86C1',fg='#FFFFFF')
text1.grid(column=2, row=11,padx=20, pady=20)
var = IntVar()
CB = Checkbutton(main, text="check this please if you want to add bias", variable=var)
CB.grid(column=2, row=12)
button = Button(main, text="submit", bg='#5DADE2',fg='#FFFFFF', command=check_bias)
button.grid(column=2, row=13)



def Task3():
    classifier = int(DS.get())
    nLayers = int(HL.get())
    nLayers += 1  # for output layer that contain either 3 or 10 neurons
    neurons = neurons_NO.get()
    nEpochs = int(EN.get())
    activ = int(AF.get())
    alpha = float(LR.get())
    bias_flag = int(var.get())
    layers_dims = neurons.split(" ")
    layers_dims = list(map(int, layers_dims))

    if classifier == 0:
        x_train, y_train, x_test, y_test = m.penguins_classification(layers_dims)
    else:
        x_train, y_train, x_test, y_test = m.digits_classification(layers_dims)

    params = m.parameters_initialization(layers_dims)
    # call gradient descent
    CCt = Label(main, text="Cost on Train data before GD: " + str(m.compute_cost(x_train, y_train, params, nLayers, activ)))
    CCt.grid(column=1, row=20)
    params = m.gradient_decent(x_train, y_train, nEpochs, nLayers, activ, params, alpha, bias_flag)

    # calculate accuracy & confusion matrix
    matrix_test, accuracy_test = m.confusion_matrix(x_test, y_test, params, nLayers, activ, layers_dims[-1])
    matrix_train, accuracy_train = m.confusion_matrix(x_train, y_train, params, nLayers, activ, layers_dims[-1])
    CC = Label(main, text="Cost on Train data after GD: " + str( m.compute_cost(x_train, y_train, params, nLayers, activ)))
    CC.grid(column=1, row=22)

    ACC1 = Label(main, text="Train Accuracy = " + str(round(accuracy_train, 2)) + "% " + " Test Accuracy = " + str(round(accuracy_test, 2)) + "%")
    ACC1.grid(column=1, row = 24)

    if classifier == 0:
        CM_view = Label(main, text="Confusion matrix: \n " + str(matrix_test))
        CM_view.grid(column=1, row=27)



text0 = Label(main, text="       ", bg='#AED6F1')
text0.grid(column=1, row=15)
button = Button(main, text="done", bg='#2E86C1',fg='#FFFFFF', command=Task3)
button.grid(column=1, row=16,padx=50)
main.mainloop()
