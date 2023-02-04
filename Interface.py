import PySimpleGUI as pg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import serial
import pandas
import os
import csv

# Initialize start graf
figure, ax = plt.subplots()
plt.plot(1, 1)
mpl.use("TkAgg")

ser = serial.Serial(
    # definer serial til at være et "Serial Port" objekt med de givne variable. Den åbner også den port der er blevet givet.
    port='COM5',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

TimingsVenstre = []
TimingsHoejre = []
TimingsVenstreGemt = []
TimingsHoejreGemt = []
counter = 0  # definer en variabel der goer vi kan se, hvilket besked nr. vi er noget til.
rx = ""  # initialize en tom string
tx = ""  # initialize en anden tom string.
FreqList = []
input_sent = False
event = None
value = None

idNumber = 1




# Toem vores ports buffer sådan at vi ikke kommer til at sende eller læse noget forkert
ser.reset_input_buffer()


def generateFreqList(lowerbound, upperbound):
    tempList = [0,0,0,0,0,0]
    space = (upperbound - lowerbound)/5
    for p in range(0, 6):
        tempList[p] = lowerbound + p * space
    return tempList

def GemDataCSV(idNr, timingsVenstre, timingsHoejre, FrequencyList, dictionary):
    if os.path.exists("HoereData.csv"):
        tempData = csv.DictReader(open("HoereData.csv", "r"))
        for line in tempData:
            print('Kører id ting')
            if line['id'] != 'id':
                if int(line['id']) > int(idNr):
                    idNr = int(line['id'])
                elif int(line['id']) == int(idNr):
                    idNr += 1

    KlikketVenstre = [2]*6
    KlikketHoejre = [2]*6
    for index, timing in enumerate(timingsVenstre):
        if timing < 0.9:
            KlikketVenstre[index] = 1
        else:
            KlikketVenstre[index] = 0
    for index, timing in enumerate(timingsHoejre):
        if timing < 0.9:
            KlikketHoejre[index] = 1
        else:
            KlikketHoejre[index] = 0


    dataline = {'id': idNr, 'Koen': dictionary['Koen'], 'Alder': dictionary['Alder'], 'Hoeretab': dictionary['Hoeretab'], 'Hovedtelefoner': dictionary['Hovedtelefoner'], 'Lokation': dictionary['Lokation'], 'Frekvens': FrequencyList, 'TimingsVenstre': timingsVenstre, 'TimingsHoejre': timingsHoejre, 'KlikketVenstre': KlikketVenstre, 'KlikketHoejre': KlikketHoejre}
    dataframe = pandas.DataFrame(dataline, columns=['id', 'Koen', 'Alder', 'Hoeretab','Hovedtelefoner', 'Lokation', 'Frekvens', 'TimingsVenstre', 'TimingsHoejre', 'KlikketVenstre', 'KlikketHoejre'])
    print(dataline)
    print(dataframe)
    if os.path.exists("HoereData.csv"):
            dataframe.to_csv("HoereData.csv", mode='a', index=False, header=False)
    else:
        dataframe.to_csv("HoereData.csv", mode='a', index=False)




InputColumn = [[pg.Text("Hvad er dit biologiske koen? Skriv 0 for kvinde og 1 for mand"), pg.InputText(key='Koen')],
               [pg.Text("Hvor gammel er du? (i hele år)"), pg.InputText(key='Alder')],
               [pg.Text("Er du blevet diagnosticeret med hoeretab? skriv 0 for nej, 1 for ja"), pg.InputText(key='Hoeretab')],
               [pg.Text("Hvor lang tid lytter du gennemsnitligt til musik i hovedtelefoner om dagen (i timer)"), pg.InputText(key='Hovedtelefoner')],
               [pg.Text("Hvor er denne proeve blevet udfoert?"), pg.InputText(key='Lokation')],
               [pg.Text("Input frekvens intervallet som testen skal udfoeres indenfor her (fx 125-8000): "), pg.InputText("125-8000", key='Frekvens')], [pg.Button("Send input")],
               [pg.Button("Luk program")]]

graphColumn = [[pg.Text("Her vil der blive vist en graf over din hoere data.")], [pg.Canvas(key="Graf")],
               [pg.Button("Opdater graf")]]

layoutColumns = [[pg.Column(InputColumn), pg.VSeparator(), pg.Column(graphColumn)]]

windowTest = pg.Window("Hoere tester", layoutColumns, location=(0, 0), finalize=True, element_justification="center")

canvas = FigureCanvasTkAgg(figure, windowTest["Graf"].TKCanvas)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)

#TEST OMRÅDE#

TestTimings1 = [0.3344, 0.3123, 0.3132, 0.12312, 0.3141, 0.4564]
TestTimings2 = [0.999, 0.3999, 0.3992, 0.99312, 0.3991, 0.4994]

# run loop
while True:
    if input_sent:
        print("Input_sent er nu True")
        ser.reset_input_buffer()
        line = ""
        while True:
            rx = str(ser.read().decode())
            if rx == ";":
                break
            else:
                line += rx
            # Print den modtagne data
        print("Fra MCU - data:", "\n------\n", line, "\n")
        if counter % 2 == 0:
            if len(TimingsVenstre) < 6:
                TimingsVenstre.append(int(line)*0.00816)
                print("Jeg har modtaget: ",line)
        elif len(TimingsHoejre) < 6:
            TimingsHoejre.append(int(line)*0.00816)
            print("Jeg har modtaget: ", line)

        counter += 1
        time.sleep(0.3)
        # clear buffer
        ser.reset_input_buffer()
    if input_sent == False:
        event, value = windowTest.read()


    if event == "Opdater graf":
        ax.cla()
        ax.grid()
        print("opdateret med", TimingsVenstre)
        print("Her er FreqList", FreqList)
        ax.plot(FreqList, TimingsVenstre, color='red', label='Venstre oere')
        ax.plot(FreqList, TimingsHoejre, color='blue', label='Hoejre oere')
        ax.set_ylabel('Reaktions tid i sekunder.')
        ax.set_xlabel('Frekvens i hz')
        ax.legend()
        ax.set_title('Reaktions tid i sekunder i forhold til frekvens i hz')
        figure.canvas.draw()
        TimingsHoejre, TimingsVenstre = [], []
        print("Burde have tegnet")

    elif event == "Send input":
        Numbers = value['Frekvens'].split('-')
        FreqList = generateFreqList(int(Numbers[0]), int(Numbers[1]))
        tx = str(Numbers[0]) + " " + str(Numbers[1]) + ";"
        ser.write(tx.encode())
        print(tx.encode())
        input_sent = True
        event = "EmptyString"

    elif event == "Luk program" or event == pg.WIN_CLOSED:
        break


    if len(TimingsHoejre) == 6:
        input_sent = False
        GemDataCSV(idNumber, TimingsVenstre, TimingsHoejre, FreqList, value)

windowTest.close()