# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:33:24 2025

@author: Josué Hernández Torres
"""
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE PARTÍCULA OSCILANTE CON CÁLCULO DE TRABAJO - PROGRAMA COMPLETO
"""
import numpy as np
import pandas as pd
import chardet
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
if 'inline' not in matplotlib.get_backend().lower():
    try:
        matplotlib.use('TkAgg')
    except:
        pass  # Backend compatible con Spyder
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

class ParticleAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Partícula Oscilante")
        self.setup_variables()
        self.create_widgets()
        
    def setup_variables(self):
        """Inicializa todas las variables de control"""
        self.file_path = tk.StringVar()
        self.dp = tk.DoubleVar(value=3e-6)
        self.dp_pix = tk.DoubleVar(value=98)
        self.fps = tk.DoubleVar(value=30)
        self.ds = tk.DoubleVar(value=1e-6)
        self.csx = tk.DoubleVar(value=988)
        self.csy = tk.DoubleVar(value=393)
        self.nfa = tk.IntVar()
        self.t_range = [None, None]
        
        self.raw_t = None
        self.raw_D = None
        self.processed_t = None
        self.processed_D = None
        
        self.last_results = {
            'time_range': "-",
            'frequency': "-",
            'mean_velocity': "-",
            'mean_acceleration': "-",
            'mean_force': "-",
            'work': "-",
            'data_points': "-"
        }

    def create_widgets(self):
        """Crea toda la interfaz gráfica"""
        # Configuración del grid principal
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Panel de control izquierdo
        control_frame = ttk.LabelFrame(main_frame, text="Parámetros", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=tk.N)
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        results_frame.grid(row=1, column=0, sticky=tk.S, pady=(10,0))
        
        # Controles de parámetros
        ttk.Label(control_frame, text="Archivo CSV:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.file_path, width=25).grid(row=0, column=1)
        ttk.Button(control_frame, text="Buscar", command=self.browse_file, width=8).grid(row=0, column=2)
        
        params = [
            ("Diámetro partícula (m):", self.dp),
            ("Diámetro partícula (px):", self.dp_pix),
            ("Framerate (fps):", self.fps),
            ("Diámetro spot (m):", self.ds),
            ("Spot X (px):", self.csx),
            ("Spot Y (px):", self.csy)
        ]
        
        for i, (text, var) in enumerate(params, start=1):
            ttk.Label(control_frame, text=text).grid(row=i, column=0, sticky=tk.W)
            ttk.Entry(control_frame, textvariable=var).grid(row=i, column=1, sticky=tk.W)
        
        ttk.Button(control_frame, text="Cargar Datos", command=self.load_data).grid(row=7, column=0, columnspan=3, pady=5)
        ttk.Button(control_frame, text="Analizar Rango", command=self.analyze_range).grid(row=8, column=0, columnspan=3, pady=5)
        ttk.Button(control_frame, text="Resetear Rango", command=self.reset_range).grid(row=9, column=0, columnspan=3, pady=5)
        
        # Panel de resultados
        self.result_labels = {}
        results = [
            ("Rango analizado:", 'time_range'),
            ("Frecuencia (Hz):", 'frequency'),
            ("Velocidad prom. (m/s):", 'mean_velocity'),
            ("Aceleración prom. (m/s²):", 'mean_acceleration'),
            ("Fuerza prom. (N):", 'mean_force'),
            ("Trabajo realizado (J):", 'work'),
            ("Puntos de datos:", 'data_points')
        ]
        
        for i, (text, key) in enumerate(results):
            ttk.Label(results_frame, text=text).grid(row=i, column=0, sticky=tk.W)
            self.result_labels[key] = ttk.Label(results_frame, text="-", font=('TkDefaultFont', 9, 'bold'))
            self.result_labels[key].grid(row=i, column=1, sticky=tk.W)
        
        # Área de gráfico interactivo
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10,0))
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)
        
        # Barra de herramientas
        self.toolbar_frame = ttk.Frame(self.graph_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky=tk.W)
        
        # Figura principal
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Barra de navegación
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky=tk.W)
        
        # Barra de estado
        self.status_bar = ttk.Label(main_frame, text="Esperando archivo...", relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def browse_file(self):
        """Selecciona archivo CSV"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.file_path.set(filename)
            try:
                with open(filename, 'rb') as f:
                    result = chardet.detect(f.read(10000))
                    f.seek(0)
                    self.nfa.set(sum(1 for _ in f) - 1)
                self.status_bar.config(text=f"Archivo cargado: {self.nfa.get()} líneas detectadas")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo leer el archivo: {str(e)}")

    def load_data(self):
        """Carga y procesa los datos iniciales"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Seleccione un archivo CSV")
            return
            
        try:
            with open(self.file_path.get(), 'rb') as f:
                result = chardet.detect(f.read(10000))
            
            df = pd.read_csv(
                self.file_path.get(), 
                encoding=result['encoding'], 
                usecols=['X','Y'],
                nrows=min(self.nfa.get(), 10000)
            )
            
            # Procesar datos
            x_part = df['X'].values
            y_part = df['Y'].values
            
            escala = self.dp.get()/self.dp_pix.get()
            Xp = (x_part - self.csx.get()) * escala
            Yp = (-y_part + self.csy.get()) * escala
            
            self.raw_D = np.sqrt(Xp**2 + Yp**2)
            n = len(Xp)
            dt = 1/self.fps.get()
            self.raw_t = np.arange(0, n*dt, dt)
            
            # Inicializar con todos los datos
            self.processed_t = self.raw_t.copy()
            self.processed_D = self.raw_D.copy()
            self.t_range = [None, None]
            
            # Graficar datos iniciales
            self.plot_raw_data()
            
            # Habilitar selección de rango
            self.span = SpanSelector(
                self.ax,
                self.on_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.5, facecolor='yellow'),
                interactive=True
            )
            
            self.status_bar.config(text="Datos cargados. Seleccione un rango con el mouse.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos:\n{str(e)}")

    def plot_raw_data(self):
        """Grafica los datos crudos"""
        self.ax.clear()
        self.ax.plot(self.raw_t, self.raw_D, 'b-', label='Datos completos')
        
        if self.t_range[0] is not None and self.t_range[1] is not None:
            self.ax.axvspan(self.t_range[0], self.t_range[1], color='yellow', alpha=0.3)
            self.ax.plot(self.processed_t, self.processed_D, 'r-', linewidth=2, label='Rango seleccionado')
        
        self.ax.set_xlabel('Tiempo (s)')
        self.ax.set_ylabel('Distancia (m)')
        self.ax.set_title('Seleccione el rango a analizar')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def on_select(self, tmin, tmax):
        """Maneja la selección de rango"""
        self.t_range = [tmin, tmax]
        mask = (self.raw_t >= tmin) & (self.raw_t <= tmax)
        self.processed_t = self.raw_t[mask]
        self.processed_D = self.raw_D[mask]
        self.plot_raw_data()

    def reset_range(self):
        """Restablece el rango seleccionado"""
        self.t_range = [None, None]
        self.processed_t = self.raw_t.copy()
        self.processed_D = self.raw_D.copy()
        self.plot_raw_data()
        self.status_bar.config(text="Rango reseteado a todos los datos.")

    def update_results_panel(self, results):
        """Actualiza el panel de resultados"""
        for key, value in results.items():
            self.result_labels[key].config(text=str(value))

    def calculate_work(self):
        """Calcula el trabajo realizado por la partícula"""
        try:
            dt = 1/self.fps.get()
            D = self.processed_D
            t = self.processed_t
            
            # Cálculos físicos
            v = np.diff(D)/dt
            a = np.diff(v)/dt
            
            # Cálculo de masa
            dp = self.dp.get()
            Volumen_sio2 = 4/3 * np.pi * ((dp/2)**3)
            Volumen_ti_max = 2 * np.pi * ((dp/2)**2) * 15e-9
            densidad_ti = 4507
            densidad_sio2 = 2634
            m = (Volumen_sio2 * densidad_sio2) + (Volumen_ti_max * densidad_ti)
            F = m * a
            
            # Desplazamientos (dr = ΔD)
            dr = np.diff(D[1:])  # Coincide con longitud de F
            
            # Cálculo del trabajo (método trapezoidal)
            #work = np.trapezoid(F, x=D[2:])  # Integración con posiciones reales
            work=np.sum(F*dr)
            # Trabajo acumulado para gráfico
            work_accumulated = np.cumsum(F * dr)
            
            return {
                'work': work,
                'work_total': work,  # Añadir esta línea
                'work_accumulated': work_accumulated,
                'force': F,
                'time_points': t[2:],
                'displacement': D[2:]
            }
            
        except Exception as e:
            raise ValueError(f"Error calculando trabajo: {str(e)}")

    def plot_work_results(self, work_data):
        try:
            # Crear figura explícitamente
            fig = plt.figure('Trabajo vs Tiempo', figsize=(10, 6))
            plt.clf()  # Limpiar figura existente
            
            ax = fig.add_subplot(111)
            
            # Verificar y preparar datos
            if len(work_data['time_points']) != len(work_data['work_accumulated']):
                raise ValueError("Datos inconsistentes")
                
            # Gráfico principal
            ax.plot(work_data['time_points'], work_data['work_accumulated'],
                   color='blue', linewidth=2, label='Trabajo acumulado')
            
            # Línea de trabajo total
            total_work = work_data.get('work_total', work_data['work_accumulated'][-1])
            ax.axhline(y=total_work, color='red', linestyle='--',
                      label=f'Trabajo total: {total_work:.2e} J')
            
            # Configuración del gráfico
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Trabajo (J)')
            ax.set_title('Trabajo Acumulado vs Tiempo')
            ax.grid(True)
            ax.legend()
            
            # Formato científico para ejes
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
            
            fig.tight_layout()
            self.canvas.draw()  # Usar el canvas de Tkinter en lugar de plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo graficar: {str(e)}")

    def plot_analysis_results(self, t, D, v, a, dp, ds, frecuencia):
        # Configurar estilo de los gráficos
        available_styles = plt.style.available
        preferred_styles = ['seaborn-v0_8', 'seaborn', 'ggplot', 'classic']
        selected_style = next((s for s in preferred_styles if s in available_styles), 'default')
        plt.style.use(selected_style)
        
        # Crear figura con subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Análisis de Partícula Oscilante', fontsize=14)
        
        # 1. Gráfico de Distancia
        ax1.plot(t[:len(D)], D, marker='.', markersize=5, linestyle='-', color='black', label='Distancia')
        ax1.axhspan(0, dp/2, facecolor='gray', alpha=0.3)
        ax1.axhspan(0, ds/2, facecolor='red', alpha=0.3)
        ax1.text(0.8*max(t), 0.6*dp, r'$r = %.1f \times 10^{-6}\,m$' % (dp/2*1e6), 
                color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        #ax1.text(0.8*max(t), 0.8*max(D), f'$f = {frecuencia:.2f}\,Hz$', 
                #color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax1.set_yticks([0, dp/2])
        ax1.set_yticklabels(['0', 'r'])
        ax1.set_ylabel("Distancia (m)")
        ax1.legend()
        ax1.grid(True)
        
        # 2. Gráfico de Velocidad - CORRECCIÓN PRINCIPAL
        ax2.plot(t[:len(v)], v, marker='.', markersize=5, linestyle='-', color='blue', label='Velocidad')
        
        # Configurar el formateador para mostrar mejor los valores
        def format_func(value, tick_number):
            return f"{value:.1f}"
        
        # Usar el formateador personalizado o científico según la magnitud
        if max(abs(v)) < 1e-1 or max(abs(v)) > 1e1:
            # Para valores muy grandes o pequeños, usar notación científica
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Ajustar estos límites según necesidad
            ax2.yaxis.set_major_formatter(formatter)
        else:
            # Para valores intermedios, usar formato decimal normal
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        ax2.set_ylabel('Velocidad (m/s)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Gráfico de Aceleración
        ax3.plot(t[:len(a)], a, marker='.', markersize=5, linestyle='-', color='red', label='Aceleración')
        
        # Aplicar misma lógica de formato que para velocidad
        if max(abs(a)) < 1e-1 or max(abs(a)) > 1e1:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            ax3.yaxis.set_major_formatter(formatter)
        else:
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Aceleración (m/s²)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)

    def analyze_range(self):
        """Realiza el análisis completo del rango seleccionado"""
        if self.raw_t is None or self.raw_D is None:
            messagebox.showerror("Error", "Primero cargue los datos")
            return
            
        try:
            if self.processed_t is None or self.processed_D is None:
                self.processed_t = self.raw_t.copy()
                self.processed_D = self.raw_D.copy()
            
            t = self.processed_t
            D = self.processed_D
            
            # Cálculos físicos básicos
            dt = 1/self.fps.get()
            v = np.diff(D)/dt
            a = np.diff(v)/dt
            
            # Cálculo de masa y fuerza
            dp = self.dp.get()
            Volumen_sio2 = 4/3 * np.pi * ((dp/2)**3)
            Volumen_ti_max = 2 * np.pi * ((dp/2)**2) * 15e-9
            densidad_ti = 4507
            densidad_sio2 = 2634
            m = (Volumen_sio2 * densidad_sio2) + (Volumen_ti_max * densidad_ti)
            F = m * a

            # Detección de frecuencia
            peaks, _ = find_peaks(D, height=0)
            if len(peaks) > 1:
                periodos = np.diff(t[peaks])
                frecuencia = 1 / np.mean(periodos)
            else:
                frecuencia = 0
                messagebox.showwarning("Advertencia", "No se detectaron suficientes picos para calcular frecuencia")

            # Cálculo del trabajo
            work_results = self.calculate_work()
            
            # Actualizar resultados
            self.last_results = {
                'time_range': f"{t[0]:.2f}s a {t[-1]:.2f}s",
                'frequency': f"{frecuencia:.4f}Hz",
                'mean_velocity': f"{np.mean(v):.4e}m/s",
                'mean_acceleration': f"{np.mean(a):.4e}m/s^2",
                'mean_force': f"{np.mean(F):.4e}N",
                'work': f"{work_results['work']:.4e}J",
                'data_points': len(t)
            }
            
            self.update_results_panel(self.last_results)
            
            # Mostrar gráficos
            self.plot_work_results(work_results)
            self.plot_analysis_results(t, D, v, a, dp, self.ds.get(), frecuencia)
            
            # Mostrar resumen
            messagebox.showinfo(
                "Resultados", 
                "\n".join(f"{k}: {v}" for k, v in self.last_results.items())
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleAnalysisApp(root)
    root.mainloop()