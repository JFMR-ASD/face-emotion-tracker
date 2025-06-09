import socket
import struct
import pickle
import mediapipe as mp
import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

# ------------------- VARIABLES GLOBALES ----------------------
running = True
analizando = False
emotion_history = []
frame_actual = None
lock = threading.Lock()
canvas_grafica = None
inicio_tiempo = 0

# ------------------- FUNCIONES -------------------------------
def cerrar():
    global running
    running = False
    root.quit()

def iniciar_analisis():
    global analizando, inicio_tiempo

    nombre = entry_nombre.get().strip()
    edad = entry_edad.get().strip()
    genero = combo_genero.get()

    if not nombre or not edad or not genero:
        lbl_estado.config(text="Completa todos los campos antes de iniciar.", fg="red")
        return
    if not edad.isdigit():
        lbl_estado.config(text="La edad debe ser un número.", fg="red")
        return

    analizando = True
    inicio_tiempo = time.time()
    lbl_estado.config(text="Análisis en curso...", fg="green")
    entry_nombre.config(state="disabled")
    entry_edad.config(state="disabled")
    combo_genero.config(state="disabled")

def detener_analisis():
    global analizando
    analizando = False
    entry_nombre.config(state="normal")
    entry_edad.config(state="normal")
    combo_genero.config(state="readonly")
    lbl_estado.config(text="Análisis finalizado.", fg="blue")
    mostrar_reporte()

def mostrar_reporte():
    global canvas_grafica
    ahora = time.time()
    recientes = [e for t, e in emotion_history if ahora - t <= 60]

    if not recientes:
        reporte_texto.delete("1.0", tk.END)
        reporte_texto.insert(tk.END, "No hay datos suficientes.\n")
        return

    acumuladas = defaultdict(float)
    for emociones in recientes:
        for k, v in emociones.items():
            acumuladas[k] += v

    num_muestras = len(recientes)
    promedios = {k: v / num_muestras for k, v in acumuladas.items()}

    texto_reporte = ""
    for k, v in sorted(promedios.items(), key=lambda item: -item[1]):
        texto_reporte += f"{k:<12}: {v:.2f}%\n"

    reporte_texto.config(state='normal')        
    reporte_texto.delete("1.0", tk.END)
    reporte_texto.insert(tk.END, texto_reporte)
    reporte_texto.config(state='disabled')       

    nombre = entry_nombre.get().strip()
    edad = entry_edad.get().strip()
    genero = combo_genero.get()
    fecha = time.strftime("%Y-%m-%d %H:%M:%S")
    nombre_archivo = f"reporte_emociones_{int(time.time())}.txt"

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write("Reporte de Análisis de Emociones\n")
        f.write("=================================\n\n")
        f.write(f"Fecha        : {fecha}\n")
        f.write(f"Nombre       : {nombre}\n")
        f.write(f"Edad         : {edad}\n")
        f.write(f"Género       : {genero}\n\n")
        f.write("Promedio de emociones (últimos 60 segundos):\n")
        f.write("-------------------------------------------\n")
        f.write(texto_reporte)


    fig, ax = plt.subplots(figsize=(6, 4), dpi=100) 
    emociones = list(promedios.keys())
    valores = list(promedios.values())
    ax.barh(emociones, valores, color='cornflowerblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje")
    ax.set_title("Promedio emociones (últimos 60s)")

    fig.subplots_adjust(left=0.3) 
    ax.set_yticklabels(emociones, fontsize=10)

    if canvas_grafica:
        canvas_grafica.get_tk_widget().destroy()

    canvas_grafica = FigureCanvasTkAgg(fig, master=grafica_frame)
    canvas_grafica.draw()
    canvas_grafica.get_tk_widget().pack()

def actualizar_barra_progreso():
    global inicio_tiempo, analizando
    if not analizando:
        progreso_bar.delete("all")
        progreso_bar.create_rectangle(0, 0, 0, 30, fill="gray")
        return

    tiempo_transcurrido = time.time() - inicio_tiempo
    progreso = min(tiempo_transcurrido / 60, 1.0)
    progreso_bar.delete("all")
    progreso_bar.create_rectangle(0, 0, progreso * 400, 30, fill="green")

    if progreso >= 1.0:
        detener_analisis()

def actualizar_frame_gui():
    global frame_actual

    if not running:
        return

    with lock:
        frame = frame_actual.copy() if frame_actual is not None else None

    if frame is None:
        video_label.after(10, actualizar_frame_gui)
        return

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        emotions = {}
        emociones_traducidas_resultado = {}
        if analizando:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emociones = result[0]['emotion']
                emociones_traducidas = {
                    'angry': 'Enojo',
                    'disgust': 'Asco',
                    'fear': 'Miedo',
                    'happy': 'Felicidad',
                    'sad': 'Tristeza',
                    'surprise': 'Sorpresa',
                    'neutral': 'Neutral'
                }
                emociones_traducidas_resultado = {emociones_traducidas[k]: v for k, v in emociones.items()}
                emotion_history.append((time.time(), emociones_traducidas_resultado))
                emotion_history[:] = [(t, e) for t, e in emotion_history if time.time() - t <= 60]
            except Exception as e:
                print(f"Error de análisis: {e}")


        y_offset = 20
        for line in ', '.join([f"{k}: {v:.1f}%" for k, v in emociones_traducidas_resultado.items()]).split(', '):
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 20


    except Exception as e:
        print("Error de análisis:", e)

    actualizar_barra_progreso()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, actualizar_frame_gui)

def recibir_frames():
    global frame_actual
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(1)
    print("Esperando conexión del cliente...")

    conn, addr = server_socket.accept()
    print("Conexión desde:", addr)

    payload_size = struct.calcsize(">L")
    data = b''

    while running:
        try:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            with lock:
                frame_actual = frame

        except Exception as e:
            print("Error de red:", e)
            break

    conn.close()
    server_socket.close()

# ------------------- LOGIN ----------------------
def verificar_login(event=None):
    usuario = entry_usuario.get()
    contrasena = entry_contrasena.get()
    if usuario == "admin" and contrasena == "1234":
        frame_login.destroy()
        inicializar_interfaz_principal()
    else:
        lbl_mensaje.config(text="Usuario o contraseña incorrectos", fg="red")

def crear_login():
    global frame_login, entry_usuario, entry_contrasena, lbl_mensaje
    color_fondo = root.cget("bg") 

    frame_login = tk.Frame(root, bg=color_fondo, padx=40, pady=40)
    frame_login.pack(expand=True)

    tk.Label(frame_login, text="Inicio de Sesión", font=("Arial", 18, "bold"), bg=color_fondo, fg="black").pack(pady=10)
    tk.Label(frame_login, text="Usuario:", font=("Arial", 14), bg=color_fondo, fg="black").pack(pady=5)
    entry_usuario = tk.Entry(frame_login, font=("Arial", 14))
    entry_usuario.pack(pady=5)

    tk.Label(frame_login, text="Contraseña:", font=("Arial", 14), bg=color_fondo, fg="black").pack(pady=5)
    entry_contrasena = tk.Entry(frame_login, show="*", font=("Arial", 14))
    entry_contrasena.pack(pady=5)

    btn_login = tk.Button(frame_login, text="Ingresar", command=verificar_login, font=("Arial", 14),
                          bg="#4CAF50", fg="white", width=20, activebackground="#45a049")
    btn_login.pack(pady=15)

    lbl_mensaje = tk.Label(frame_login, text="", fg="red", font=("Arial", 12), bg=color_fondo)
    lbl_mensaje.pack(pady=5)


    entry_usuario.bind("<Return>", verificar_login)
    entry_contrasena.bind("<Return>", verificar_login)

# ------------------- INTERFAZ PRINCIPAL ----------------------
def inicializar_interfaz_principal():
    global video_label, progreso_bar, reporte_texto, grafica_frame
    global entry_nombre, entry_edad, combo_genero, lbl_estado

    def crear_boton_con_estilo(parent, texto, comando):
        btn = tk.Button(parent, text=texto, command=comando, **estilo_btn)
        btn.pack(side="left", padx=5)

        def on_enter(e): btn.config(bg="#289df5")
        def on_leave(e): btn.config(bg="#007ACC")
        def on_press(e): btn.config(bg="#005f99")
        def on_release(e): btn.config(bg="#289df5")

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<ButtonPress-1>", on_press)
        btn.bind("<ButtonRelease-1>", on_release)

        return btn

    frame_izquierdo = tk.Frame(root, padx=20, pady=10, bg="#1e1e1e")
    frame_izquierdo.pack(side="left", fill="both", expand=True)

    frame_derecho = tk.Frame(root, padx=20, pady=10, bg="#2e2e2e")
    frame_derecho.pack(side="right", fill="y")

    video_label = tk.Label(frame_izquierdo, bd=2, relief="groove", bg="#1e1e1e", fg="white")
    video_label.pack(pady=10)

    frame_datos = tk.LabelFrame(frame_izquierdo, text="Datos del Usuario", font=("Arial", 12, "bold"), bg="#1e1e1e", fg="white")
    frame_datos.pack(pady=10, fill="x")

    tk.Label(frame_datos, text="Nombre:", font=("Arial", 12), bg="#1e1e1e", fg="white").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    entry_nombre = tk.Entry(frame_datos, font=("Arial", 12), bg="#2e2e2e", fg="white", insertbackground="white")
    entry_nombre.grid(row=0, column=1, padx=5)

    tk.Label(frame_datos, text="Edad:", font=("Arial", 12), bg="#1e1e1e", fg="white").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    entry_edad = tk.Entry(frame_datos, font=("Arial", 12), bg="#2e2e2e", fg="white", insertbackground="white")
    entry_edad.grid(row=1, column=1, padx=5)

    tk.Label(frame_datos, text="Género:", font=("Arial", 12), bg="#1e1e1e", fg="white").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    combo_genero = ttk.Combobox(frame_datos, values=["Masculino", "Femenino"], state="readonly", font=("Arial", 12), background="#2e2e2e", foreground="white")
    combo_genero.grid(row=2, column=1, padx=5)
    combo_genero.set("Masculino")

    lbl_estado = tk.Label(frame_datos, text="", fg="blue", bg="#1e1e1e")
    lbl_estado.grid(row=3, columnspan=2, pady=5)

    frame_botones = tk.Frame(frame_izquierdo, bg="#1e1e1e")
    frame_botones.pack(pady=10)

    estilo_btn = {"font": ("Arial", 12), "bg": "#007ACC", "fg": "white", "width": 15, "relief": "raised"}

    crear_boton_con_estilo(frame_botones, "Iniciar Análisis", iniciar_analisis)
    crear_boton_con_estilo(frame_botones, "Detener Análisis", detener_analisis)
    crear_boton_con_estilo(frame_botones, "Cerrar", cerrar)

    frame_progreso = tk.Frame(frame_izquierdo, bg="#1e1e1e")
    frame_progreso.pack(pady=10)
    progreso_bar = tk.Canvas(frame_progreso, width=400, height=30, bg="lightgray")
    progreso_bar.pack()

    reporte_label = tk.Label(frame_derecho, text="Estadísticas (últimos 60s):", font=('Arial', 12, 'bold'), bg="#2e2e2e", fg="white")
    reporte_label.pack()

    reporte_texto = tk.Text(
        frame_derecho,
        height=12,
        width=40,
        font=('Courier', 10),
        bd=2,
        relief="sunken",
        bg="#2e2e2e",
        fg="white",
        state='disabled'
    )
    reporte_texto.pack(pady=5)

    grafica_frame = tk.Frame(frame_derecho, bg="#2e2e2e")
    grafica_frame.pack(pady=10)

    global mp_face_mesh, face_mesh, mp_drawing, mp_drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    threading.Thread(target=recibir_frames, daemon=True).start()
    actualizar_frame_gui()



# ------------------- INICIO ----------------------
root = tk.Tk()
root.title("FaceTracker")
root.state('zoomed')
crear_login()
root.mainloop()
cv2.destroyAllWindows()
