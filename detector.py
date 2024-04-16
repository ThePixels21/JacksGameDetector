from customtkinter import CTk, CTkFrame, CTkButton, CTkCheckBox
from tkinter import IntVar, Label, filedialog, StringVar, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import webbrowser
import cv2

# By @ThePixels21

model = YOLO('bestv8.pt')  # Modelo entrenado

rangos_colores = {
    'Verde': [np.array([36, 100, 20], np.uint8), np.array([70, 255, 255], np.uint8)],
    'Rosado': [np.array([160, 100, 100], np.uint8), np.array([170, 255, 255], np.uint8)],
    'Amarillo': [np.array([28, 150, 20], np.uint8), np.array([32, 255, 255], np.uint8)],
    'Naranja': [np.array([7, 200, 20], np.uint8), np.array([18, 255, 255], np.uint8)]
}


def deteccion_tiempo_real():
    # Código para la detección en tiempo real...
    cap = cv2.VideoCapture(int(camNumber.get()))

    while True:

        ret, frame = cap.read()

        if var.get() == 1:
            # Invertir el frame
            frame = cv2.flip(frame, 1)

        # Realizar detección
        resultados = model.predict(frame, imgsz=640, conf=0.35)

        # Inicializar contadores de colores y clases
        count_colores = {'Verde': 0, 'Rosado': 0, 'Amarillo': 0, 'Naranja': 0}
        count_clases = {'Bola': 0, 'Ficha': 0}

        # Filtrar detecciones por tamaño
        info_filtrada = []
        for r in resultados:
            for box in r.boxes:
                xi, yi, xf, yf = box.xyxy[0].tolist()  # obtener las coordenadas de la caja
                # Calcular el área de la detección
                area = (xf - xi) * (yf - yi)
                # Si el área es menor que el umbral, agregar la detección a la lista filtrada
                if area < 35000:
                    info_filtrada.append(box)
                    # Dibujar el rectángulo de la detección en el frame
                    if model.names[int(box.cls)] == 'Bola':
                        color = (0, 0, 255)  # Rojo
                        count_clases['Bola'] += 1
                    elif model.names[int(box.cls)] == 'Ficha':
                        color = (255, 0, 0)  # Azul
                        count_clases['Ficha'] += 1
                    else:
                        color = (0, 255, 0)  # Verde
                    cv2.rectangle(frame, (int(xi), int(yi)), (int(xf), int(yf)), color, 2)
                    # Dibujar el nombre de la clase sobre el rectángulo
                    cv2.putText(frame, model.names[int(box.cls)], (int(xi), int(yi) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, color, 2)
                    # Aumentar el contador del color correspondiente
                    roi = frame[int(yi):int(yf), int(xi):int(xf)]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    max_count = 0
                    max_color = None
                    for color_name, (bajo, alto) in rangos_colores.items():
                        mask = cv2.inRange(hsv, bajo, alto)
                        non_zero_count = cv2.countNonZero(mask)
                        if non_zero_count > max_count:
                            max_count = non_zero_count
                            max_color = color_name

                    if max_color is not None:
                        count_colores[max_color] += 1

        # Cambiar el tamaño del frame
        frame = cv2.resize(frame, (800, 600))

        # Mostrar los contadores de colores en la parte superior izquierda
        y = 30
        for color_name, count in count_colores.items():
            color = None
            if color_name == 'Verde':
                color = (0, 255, 0)
            elif color_name == 'Rosado':
                color = (255, 0, 255)
            elif color_name == 'Amarillo':
                color = (0, 255, 255)
            elif color_name == 'Naranja':
                color = (0, 165, 255)
            cv2.putText(frame, '{}: {}'.format(color_name, count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                        cv2.LINE_AA)
            y += 30

        # Mostrar los contadores de clases en la parte inferior izquierda
        y = frame.shape[0] - 60
        for class_name, count in count_clases.items():
            color = None
            if class_name == 'Bola':
                color = (0, 0, 255)  # Rojo
            elif class_name == 'Ficha':
                color = (255, 0, 0)  # Azul
            cv2.putText(frame, '{}: {}'.format(class_name + 's', count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
                        2, cv2.LINE_AA)
            y += 30

        # Mostrar FPS
        cv2.imshow('Detector de fichas', frame)

        # Leer el teclado
        t = cv2.waitKey(5)
        try:
            if t == 27 or cv2.getWindowProperty('Detector de fichas', 0) < 0:
                break
        except cv2.error as e:
            break

    cap.release()
    cv2.destroyAllWindows()


def deteccion_imagen(ruta_imagen=None):
    # Abre el explorador de archivos para seleccionar una imagen
    if ruta_imagen is None:
        root = CTk()
        ruta_imagen = filedialog.askopenfilename(initialdir="/", title="Selecciona una imagen",
                                                 filetypes=(
                                                 ("png files", "*.png"), ("jpg files", "*.jpg"), ("all files", "*.*")))
        root.destroy()

    # Carga la imagen
    img = cv2.imread(ruta_imagen)

    # Realizar detección
    resultados = model.predict(img, imgsz=640, conf=0.25)

    # Inicializar contadores de colores y clases
    count_colores = {'Verde': 0, 'Rosado': 0, 'Amarillo': 0, 'Naranja': 0}
    count_clases = {'Bola': 0, 'Ficha': 0}

    # Filtrar detecciones por tamaño
    info_filtrada = []
    for r in resultados:
        for box in r.boxes:
            xi, yi, xf, yf = box.xyxy[0].tolist()  # obtener las coordenadas de la caja
            # Calcular el área de la detección
            area = (xf - xi) * (yf - yi)
            # Si el área es menor que un cierto umbral, agregar la detección a la lista filtrada
            if area < 50000:
                info_filtrada.append(box)
                # Dibujar el rectángulo de la detección en el frame
                if model.names[int(box.cls)] == 'Bola':
                    color = (0, 0, 255)  # Rojo
                    count_clases['Bola'] += 1
                elif model.names[int(box.cls)] == 'Ficha':
                    color = (255, 0, 0)  # Azul
                    count_clases['Ficha'] += 1
                else:
                    color = (0, 255, 0)  # Verde
                cv2.rectangle(img, (int(xi), int(yi)), (int(xf), int(yf)), color, 2)
                # Dibujar el nombre de la clase sobre el rectángulo
                cv2.putText(img, model.names[int(box.cls)], (int(xi), int(yi) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)
                # Aumentar el contador del color correspondiente
                roi = img[int(yi):int(yf), int(xi):int(xf)]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                max_count = 0
                max_color = None
                for color_name, (bajo, alto) in rangos_colores.items():
                    mask = cv2.inRange(hsv, bajo, alto)
                    non_zero_count = cv2.countNonZero(mask)
                    if non_zero_count > max_count:
                        max_count = non_zero_count
                        max_color = color_name

                if max_color is not None:
                    count_colores[max_color] += 1

    # Cambiar el tamaño del frame
    img = cv2.resize(img, (800, 600))

    # Mostrar los contadores de colores en la parte superior izquierda
    y = 30
    for color_name, count in count_colores.items():
        color = None
        if color_name == 'Verde':
            color = (0, 255, 0)
        elif color_name == 'Rosado':
            color = (255, 0, 255)
        elif color_name == 'Amarillo':
            color = (0, 255, 255)
        elif color_name == 'Naranja':
            color = (0, 165, 255)
        cv2.putText(img, '{}: {}'.format(color_name, count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
                    cv2.LINE_AA)
        y += 30

    # Mostrar los contadores de clases en la parte inferior izquierda
    y = img.shape[0] - 60
    for class_name, count in count_clases.items():
        color = None
        if class_name == 'Bola':
            color = (0, 0, 255)  # Rojo
        elif class_name == 'Ficha':
            color = (255, 0, 0)  # Azul
        cv2.putText(img, '{}: {}'.format(class_name + 's', count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
                    2, cv2.LINE_AA)
        y += 30

    # Mostrar la imagen con las detecciones
    cv2.imshow('Detector de fichas', img)
    while True:
        try:
            if cv2.getWindowProperty('Detector de fichas', 0) >= 0:  # Verificar si la ventana está abierta
                if cv2.waitKey(5) & 0xFF == 27:  # 27 es el código ASCII para la tecla Escape
                    break
            else:
                break
        except cv2.error as e:
            break
    cv2.destroyAllWindows()


def tomar_foto():
    # Abre la cámara
    cap = cv2.VideoCapture(int(camNumber.get()))

    while True:
        # Toma un frame de la cámara
        ret, frame = cap.read()

        if var.get() == 1:
            # Invertir el frame
            frame = cv2.flip(frame, 1)

        # Muestra el frame en una ventana
        cv2.imshow('Presiona la tecla "s" para tomar una foto', frame)

        # Espera a que se presione una tecla
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Si se presionó la tecla "s", guarda la imagen
            cv2.imwrite('foto.png', frame)
            break

    # Libera la cámara y cierra las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()

    # Llama a la función de detección en la imagen que acabas de tomar
    deteccion_imagen('foto.png')


def detect_cameras(max_cameras=10):
    cameras = []
    for i in range(max_cameras):
        temp_camera = cv2.VideoCapture(i)
        if temp_camera.isOpened():
            cameras.append(i)
            temp_camera.release()
    return cameras


def open_github(event):
    webbrowser.open('https://github.com/ThePixels21')


# Crea la ventana con los botones
root = CTk()

# Configura la geometría de la ventana
root.geometry("480x600+350+20")
root.minsize(480, 600)
root.config(bg='#010101')
root.iconbitmap('logo.ico')
root.title("Detección de Jacks")

frame = CTkFrame(root, fg_color='#010101')
frame.grid(column=0, row=0, sticky='nsew', padx=50, pady=50)

frame.columnconfigure([0, 1], weight=1)
frame.rowconfigure([0, 1, 2, 3, 4, 5], weight=1)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Carga el logo en formato PNG con ImageTk
logo = Image.open('logo.png')
logo = logo.resize((300, 300))  # Ajusta el tamaño a tus necesidades
logo = ImageTk.PhotoImage(logo)

# Mantén una referencia a la imagen para evitar que sea eliminada por el recolector de basura
root.logo = logo

Label(frame, image=root.logo, bg='#010101').grid(columnspan=2, row=0)

# Ajusta los colores y estilos de los botones para que se parezcan más a los del proyecto que compartiste
CTkButton(frame, text="Detección en tiempo real", command=deteccion_tiempo_real, bg_color='#010101', fg_color='#2cb67d',
          hover_color='#7f5af0', corner_radius=12, border_width=2).grid(columnspan=2, row=1, padx=4, pady=4)
CTkButton(frame, text="Tomar foto", command=tomar_foto, bg_color='#010101', fg_color='#2cb67d', hover_color='#7f5af0',
          corner_radius=12, border_width=2).grid(columnspan=2, row=2, padx=4, pady=4)
CTkButton(frame, text="Seleccionar imagen", command=deteccion_imagen, bg_color='#010101', fg_color='#2cb67d',
          hover_color='#7f5af0', corner_radius=12, border_width=2).grid(columnspan=2, row=3, padx=4, pady=4)

# Agrega un checkmark para el "Modo espejo"
var = IntVar()
CTkCheckBox(frame, text="Modo espejo", variable=var, bg_color='#010101', fg_color='#2cb67d', hover_color='#7f5af0',
            border_color='#2cb67d').grid(columnspan=2, row=4, padx=4, pady=4)

# Agrega un Label para indicar que el Combobox se utiliza para seleccionar el número de cámara
camNumberLabel = Label(frame, text="Número de cámara:", bg='#010101', fg='#2cb67d')
camNumberLabel.grid(column=0, row=5, padx=4, pady=4, sticky='e')

# Agrega un Combobox para seleccionar el número de cámara
camNumber = StringVar()
camNumberCombobox = ttk.Combobox(frame, textvariable=camNumber, background='#010101', foreground='#2cb67d')
camNumberCombobox['values'] = detect_cameras()  # Detecta las cámaras disponibles
camNumberCombobox.grid(column=1, row=5, padx=4, pady=4, sticky='w')
camNumberCombobox.current(0)  # Establece el valor inicial en 0

# Personaliza el aspecto del Combobox
style = ttk.Style()
style.theme_use('alt')
style.configure("TCombobox", fieldbackground='#010101', background='#010101', foreground='#2cb67d')
style.map('TCombobox', selectbackground=[('readonly', '#010101')], selectforeground=[('readonly', '#2cb67d')])

developerLabel = Label(root, text="By @ThePixels21", bg='#010101', fg='#2cb67d', cursor="hand2")
developerLabel.place(relx=1.0, rely=1.0, x=-5, y=-5, anchor='se')
developerLabel.bind("<Button-1>", open_github)

root.mainloop()