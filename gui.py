import PySimpleGUI as sg
import os.path
import cv2
from PIL import Image
import io
import traffic_counter as counter

file_list_column = [
    [
        sg.Text("Video Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")
    ]
]

image_viewer_column = [
    [sg.Text(size=(100, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-", filename="")],
    [sg.Button("Count")],
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column([[]], key="-COLUMN-"),
        sg.Column(image_viewer_column)
    ]
]

window = sg.Window(title="Traffic Counter", layout=layout, resizable=True, finalize=True)
window.maximize()
orig_frame = []

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        files = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
        ]
        window["-FILE LIST-"].update(files)
    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT-"].update(filename)
            cap = cv2.VideoCapture(filename)
            ret, frame = cap.read()
            orig_frame = frame
            height, width = frame.shape[:2]
            img = Image.fromarray(frame)  # create PIL image from frame
            bio = io.BytesIO()  # a binary memory resident stream
            img.save(bio, format='PNG')  # save image as png to it
            imgbytes = bio.getvalue()
            window['-IMAGE-'].update(data=imgbytes)
            if not ('slider' in window.AllKeysDict):
                window.extend_layout(window['-COLUMN-'], [[sg.Slider(range=(0, height), default_value=0, size=(30, 20),
                                                                     orientation="v", enable_events=True,
                                                                     key="slider")]])
            else:
                window["slider"].update(range=(height, 0))
        except:
            pass
    elif event == "slider":
        y = values["slider"]
        frame = orig_frame.copy()
        height, width = frame.shape[:2]
        cv2.line(frame, (0, int(height - y)), (int(width), int(height - y)), (255, 0, 0), thickness=1)
        img = Image.fromarray(frame)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        imgbytes = bio.getvalue()
        window['-IMAGE-'].update(data=imgbytes)
    elif event == "Count":
        filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
        counter.call_counter_function(filename, values["slider"])
        # a 87. sor helyére kellene a forgalomszámoló
        # függvényt hívni,a a paraméterek maradjanak ugyan ezek
window.close()
