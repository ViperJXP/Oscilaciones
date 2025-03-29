# Oscilaciones
Análisis de JPs oscilantes Ópticamente Confinadas

El repositorio contiene un programa llamado "Análisis Oscilación.py", al ejecutarse se abre una ventana interactiva, en la parte superior izquierda se cargan los archivos (con el estilo o formato de los archivos *.csv que también se adjuntan en el repositorio como ejemplos), se le debe dar clic en "Cargar Datos", esto genera una vista previa de las posiciones con respecto al tiempo extraído del archivo, si el intervalo de datos se va a utilizar completo, directamente se le da clic en "Analizar Rango", de donde se extraen resultados como la frecuencia de oscilación, la velocidad promedio, la aceleración promedio, la fuerza promedio y el trabajo total realizado por la partícula, así como otro par de datos "extra" como el rango de tiempo analizado de la muestra y el número de puntos tomados para el análisis.
Si no todo el intervalo de datos sirve, sino sólo un subconjunto, entonces por medio del mouse, "subrayar" el intervalo de datos que se desee analizar.
Una vez escogido el rango y éste cambie a amarillo, entonces dar clic en "Analizar Rango".
Si hubo algún error o, simplemente, se desea analizar un intervalo diferente al seleccionado la vez anterior, dar clic en "Resetear Rango" y proceder a escoger de nuevo el intervalo que se desea analizar.

Los resultados salen en una ventana de comunicaciones, sin embargo, se plasman también en la parte inferior izquierda de la ventana donde se cargan los datos.
Otro producto es una gráfica en configuración (1,3), i.e 1 columna y 3 filas de gráficas, ordenadas de la siguiente manera:
  1.-Superior: "Distancias VS Tiempos"; Esta gráfica se crea con sólo 2 marcas en el eje y, el 0 y "r" que equivale al tamaño del radio de la partícula que se trabaja, para el que también se sombrea con una franja gris, por otro lado, la franja roja, equivale al radio del spot utilizado.  
  2.-Medio: Velocidades VS Tiempos
  3.-Inferior: Aceleraciones VS Tiempos
Finalmente, en otra ventana, se muestra una gráfica del Trabajo VS Tiempo, colocando una serie azul sólido para el trabajo y una roja punteada para el trabajo total, mostrando a su vez, el valor numérico de este trabajo total.



######################################  Futuras Optimizaciones  #####################################
1.-Las distancias están tomadas solamente en $\mathbb{R}^2$ (2 dimensiones), pero ya se está trabajando en un programa para obtener posiciones en z también, con lo que podríamos obtener posiciones en $\mathbb{R}^3$ y con ello un análisis más preciso de la dinámica de la partícula.
2.-No es optimización de este programa como tal, sino más bien de un proyecto más grande y es: que sea un módulo que analice la parte de las oscilaciones, si es el caso que este comportamiento se presenta, si no, que prescinda de él y analice lo que sea necesario para el trabajo.
