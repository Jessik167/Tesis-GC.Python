from numba import *
from numba import cuda, vectorize
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import networkx as nx
import scipy as sp
import bolsas
import Cruza
import Busquedas_locales
import random
import copy
import math
import time
import winsound

metropolis_gpu = cuda.jit(device=True)(Busquedas_locales.Busqueda_MetropolisCUDA)
Escalando_gpu = cuda.jit(device=True)(Busquedas_locales.Busqueda_EscalandoCUDA)

path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'
#Nombre_benchmark = "le450_15c"#"le450_25c"'flat300_28'"myciel3""myciel4"'2-Fullins_3'"jean""anna""DSJC250-5""DSJC500-5"

#Datos Algoritmo Genético
tam_poblacion = 128
numGeneraciones = 20
numColores = 2
semilla = 1

#Datos CUDA
bloques = 2
hilos = 32

#Datos del grafo aleatorio
numNodosAle = 100
probabilidad = 0.4
grafo = 1



'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre, nodetype=int)



'''Inicializa la población con estrategia greedy, y luego realiza una búsqueda local'''
def InicializacionPoblacion(poblacion, probabilidades, G,numNodos):# Crea instancias de colorado del grafo con estrategia greedy de tamaño de la población
    for ind in range(tam_poblacion):    #Lo realiza hasta el tamaño de la población
        if ind == 0:
            poblacion.append(TomaGreedy(G,numNodos))
        else:
            poblacion.append(bolsas.crea_individuo(G, numColores,numNodos))  #crea un individuo
        probabilidades.append(bolsas.cuenta_nodos(poblacion[ind], numColores, numNodos))    # cuenta los nodos de cada bolsa del individuo



'''Toma la solución greedy smallest last y la convierte a un individuo en diccionario'''
def TomaGreedy(G,numNodos):
    d = nx.coloring.greedy_color(G, strategy='smallest_last')   #Utiliza estrategia greedy
    Bolsas_colores = {k: {} for k in range(numColores)}
    for nodo in range(1, numNodos+1):
        if nodo in d.keys():
            if d[nodo] < numColores: #Pregunta si el color del greedy está dentro del rango de colores
                Bolsas_colores[d[nodo]][nodo] = nodo #Si si, ingresa el nodo en la bolsa que eligió el greedy
            else:
                Bolsas_colores[random.randint(0, numColores - 1)][nodo] = nodo  #Si no está, ingresa el nodo en una bolsa aleatoria
        else:
            Bolsas_colores[random.randint(0, numColores - 1)][nodo] = nodo  # Si no está, ingresa el nodo en una bolsa aleatoria
    return Bolsas_colores



'''Toma aleatoriamente dos de los indices dentro del total de la población, que serán los padres que se cruzarán'''
def EligePadres(poblacion,numNodos, padres,probabilidades):
    individuos = [] #Lista que contendrá a los dos individuos a elegir
    indices = ()

    indices += (random.randint(0, tam_poblacion - 1),)  #Elige el índice a elegir del padre 1
    while True:
        r = random.randint(0, tam_poblacion - 1)
        if r != indices[0]:
            indices += (r,) #Elige el índice a elegir del padre 2
            break
    padres.append(indices)

    individuos.append(copy.deepcopy(poblacion[padres[-1][0]]))   # toma el individuo al azar de la población (padre 1)
    individuos.append(copy.deepcopy(poblacion[padres[-1][1]]))   # toma el individuo al azar de la población (padre 2)

    return individuos



'''Recibe dos padres, los cruza con el algoritmo GPX, retorna un nuevo individuo (hijo)'''
def CruzaPadres(individuos,probabilidades,AMonoNuevo,numNodos,G):
    nuevo_indiv = {}  # Lista que contendrá al nuevo individuo
    nuevo_indiv = Cruza.GPX(individuos, numColores)  # Cruza a los padres y forma un nuevo individuo
    probabilidades.append(bolsas.cuenta_nodos(nuevo_indiv, numColores, numNodos))  # cuenta los nodos de cada bolsa del individuo

    # Calcula el número de aristas monocromáticas del individuo nuevo
    AMonoNuevo.append(Busquedas_locales.numeroAristasMono(G, nuevo_indiv,numColores))

    return nuevo_indiv



'''Cuza a los individuos, tomando aleatoriamente cada uno de los indices dentro del total de la población, 
y los cruza utilizando GPX'''
def CruzaIndividuos(poblacion,numNodos, padres,probabilidades):
    individuos = [] #Lista que contendrá a los dos individuos a elegir
    nuevos_indiv = {}  #Lista que contendrá al nuevo individuo
    indices = ()

    indices += (random.randint(0, tam_poblacion - 1),)  #Elige el índice a elegir del padre 1
    while True:
        r = random.randint(0, tam_poblacion - 1)
        if r != indices[0]:
            indices += (r,) #Elige el índice a elegir del padre 2
            break
    padres.append(indices)

    individuos.append(copy.deepcopy(poblacion[padres[-1][0]]))   # toma el individuo al azar de la población (padre 1)
    individuos.append(copy.deepcopy(poblacion[padres[-1][1]]))   # toma el individuo al azar de la población (padre 2)

    nuevos_indiv = Cruza.GPX(individuos, numColores)   # Cruza a los padres y forma un nuevo individuo
    probabilidades.append(bolsas.cuenta_nodos(nuevos_indiv, numColores, numNodos)) # cuenta los nodos de cada bolsa del individuo

    return nuevos_indiv



'''Manda a llamar a la búsqueda local en CUDA(metrópolis ó Escalando la colina)'''
@cuda.jit
def BusquedaLocal_CUDA(nuevo_individuos, probabilidades,Aristmono,MatrizAdjacencia, H,numNodos,tam_pobla,rng_states):
    bx = cuda.blockIdx.x
    thx = cuda.threadIdx.x
    bw = cuda.blockDim.x
    id = thx + bx * bw

    metropolis_gpu(MatrizAdjacencia,nuevo_individuos[id], probabilidades[id], Aristmono[id], numColores, numNodos, rng_states,id)  # regresa al individuo después de realzar la búsqueda local
    #Escalando_gpu(MatrizAdjacencia,nuevo_individuos[id], probabilidades[id], Aristmono[id], numColores, numNodos, rng_states,id)  # regresa al individuo después de realzar la búsqueda local
    cuda.syncthreads()
    #print("Sale")



'''Reemplaza al peor de los padres elegidos con el hijo, compara el número de aristas monocromáticas de los individuos implicados'''
def ActualizaPoblacion(poblacion, hijo, AristasMono, AristasMonohijo,probabilidades, probab_hijo, indiceP):
    if AristasMono[indiceP[0]] < AristasMono[indiceP[1]]: #Si el padre 2 tiene mayor número de aristas mono que el padre 1 entonces...
        poblacion[indiceP[1]]= hijo    #El hijo reemplaza al padre 2
        AristasMono[indiceP[1]]= AristasMonohijo    #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indiceP[1]]= probab_hijo
    else:                                   #Si no...
        poblacion[indiceP[0]] = hijo   #El hijo reemplaza al padre 1
        AristasMono[indiceP[0]] = AristasMonohijo   #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indiceP[0]] = probab_hijo



'''Convierte el hijo en un diccionario de colores'''
def convert_hijo(Arr,numNodos):
    Bolsas_colores = {k: {} for k in range(numColores)}
    for i in range(numColores):
        for j in range(numNodos):
            if Arr[i][j] == 1:
                Bolsas_colores[i][j+1] = j+1
    return Bolsas_colores



'''Actualiza la población por medio del arreglo numpy utilizado para CUDA'''
def ActualizaPoblacionCUDA(poblacion,hijo,AristasMono,AristasMonoHijo,probabilidades,probabilidadHijo,indicesP,numNodos):
    if AristasMono[indicesP[0]] < AristasMono[indicesP[1]]: #Si el padre 2 tiene mayor número de aristas mono que el padre 1 entonces...
        poblacion[indicesP[1]]= convert_hijo(hijo,numNodos)    #El hijo reemplaza al padre 2
        AristasMono[indicesP[1]]= AristasMonoHijo    #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indicesP[1]]= probabilidadHijo
    else:                                   #Si no...
        poblacion[indicesP[0]] = convert_hijo(hijo,numNodos)   #El hijo reemplaza al padre 1
        AristasMono[indicesP[0]] = AristasMonoHijo   #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indicesP[0]] = probabilidadHijo



'''Compara los resultados obtenidos en la generación actual y la anterior'''
def sonIguales(resultados,tam):
    primero = resultados[0]  #Toma el primer resultado
    i = 1

    for i in range(tam):    #Lo compara con todos los demas
        if primero == resultados[i]:    #Si es igual continua
            continue
        else:
            return False    #Si no, termina
    return True #Si termina el ciclo es que todos fueron iguales



'''convierte de la estructura en python a un arreglo numpy para poder ser utilizado en CUDA'''
def convierte_individuonumpy(individuo,probabilidades,Amono,indi_np,prob_np,AmonoNp,num_indiv,bolsas):
    for i in range(num_indiv):
        AmonoNp[i] = Amono[i]
        for j in range(bolsas):
            prob_np[i][j]= probabilidades[i][j]
            for r in individuo[i][j]:
                indi_np[i][j][r-1] = 1



'''Actualiza el mejor individuo encontrado con el menor número de aristas monocromáticas (NAM)'''
def ActualizaMejor(AMonocromaticas,poblacion,best,best_ind,ind,termina):
    if AMonocromaticas[ind] < best:  # Guarda al individuo que obtuvo el menor número de aristas monocromáticas
        best[0] = AMonocromaticas[ind]
        best_ind[0] = poblacion[ind]
        #return AMonocromaticas[ind], poblacion[ind]
    if AMonocromaticas[ind]==0:
        termina = True



'''Manda a llamar a la búsqueda local (metrópolis ó Escalando la colina)'''
def BusquedaLocal(nuevo_individuo, probabilidad, Aristmono, M):
     #nuevo_individuo, Arist = Busquedas_locales.Busqueda_Escalando(M, nuevo_individuo, probabilidad,
     #                                                       Aristmono, numColores)  # regresa al individuo después de realzar la búsqueda local
     nuevo_individuo, Arist = Busquedas_locales.Busqueda_Metropolis(M, nuevo_individuo, probabilidad,
                                                             Aristmono,numColores)  # regresa al individuo después de realzar la búsqueda local
     return Arist



'''Realiza una búsqueda local en la población inicial, calcula el número de aristas monocromáticas y actualiza el mejor'''
def Busqueda_Local(G,AMonocromaticas,poblacion,probabilidades,best,best_ind,termina):
    for ind in range(tam_poblacion):  # Realiza la búsqueda local en la población inicial
        AMonocromaticas.append(Busquedas_locales.numeroAristasMono(G, poblacion[ind],
                                                                   numColores))  # calcula el número de aristas monocromáticas del individuo

        ActualizaMejor(AMonocromaticas,poblacion,best,best_ind,ind,termina)

        AMonocromaticas[ind] = BusquedaLocal(poblacion[ind], probabilidades[ind], AMonocromaticas[ind], G)

        if AMonocromaticas[ind] == 0:
            best = 0
            best_ind = poblacion[ind]
            return 1
        else:
            ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
            if termina == True:
                break
    return 0


'''Si tres generaciones no encuentra nada mejor, el progama termina'''
def Termina_Generacion(bestAnt,AMonoNuevo,gen,NumGenIgual):
    bestAnt.append(AMonoNuevo)
    if (gen + 1) % NumGenIgual == 0:
        if sonIguales(bestAnt, NumGenIgual):
            bestAnt = []
            return 0
        bestAnt = []



'''Función encargada de imprimir al mejor individuo, el mejor número de aristas monocromáticas, y el tiempo total del programa'''
def imprimeResultados(best,best_ind,start_time,archivo):
    print('\n\nmejor número de aristas monocromáticas: ')
    print(str(best))
    print('mejor individuo: ')
    print(best_ind)
    print("%s seconds" % (time.time() - start_time))
    # bolsas.Muestra_Grafo(G)
    # bolsas.colorea_grafo(G,best_ind)
    # if best == 0:
    archivo.write(str(time.time() - start_time) + '\t' +str(best[0]) + '\n')
    #    archivo.write('Grafo#' + str(i + 1) + '\t' + str(numColores) + '\t' + str(time.time() - start_time) + '\n')
    # archivo.close()




#***************************************  Algoritmo Genético   *********************************************
def main():
    # =================================== Definicion de Grafos y lectura de grafos aleatorios ==================
    nombreArch = "./Resultados/TyN_" + str(probabilidad) + "_" + str(numColores) + "_" + str(tam_poblacion)
    archivo = open(nombreArch,"a")
    for i in range(10):

        #print('\n***Grafo#' + str(i + 1))
        carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
        nombre= '/G' + str(numNodosAle) + '_' + str(probabilidad) +'_' + str(grafo)#+ str(i+1)
        #nombre = '/G' + str(probabilidad) + '_' + str(i + 1)#+ str(grafo)
        Nombre_benchmark = carpeta + nombre

        #G = bolsas.crea_grafo(Nombre_benchmark) #Crea el grafo apartir de un archivo de texto
        #G=bolsas.crea_grafo_aleatorio(numNodosAle, probabilidad)   #Crea un grafo a partir de un número de nodos y una probabilidad
        G = Lee_Grafo(path + Nombre_benchmark)
        #print(Matriz_Adyacencia)
        numNodos = max(G.nodes)

        # ==========================================================================================================
        #=================================== Definicion de variables y estructuras de datos ========================
        ind = 0
        mitad_poblacion = int(tam_poblacion / 2)
        termina = False
        NumGenIgual = 3
        best= np.array([np.inf])
        bestAnt = []
        best_ind = [0]
        poblacion = []
        probabilidades=[]
        AMonocromaticas = []
        nuevos_individuos = []
        AMonoNuevo = []
        probabilidadeshijo = []
        ind_padres = []

        #Estructuras de los nuevos individuos
        nuevos_individuos = []
        AMonoNuevo = []
        ind_padres = []
        probabilidadeshijo = []
        Padres = [] #Lista que contendrá a los dos padres a elegidos
        # ==========================================================================================================
        #=================================== Definicion de variables para CUDA =====================================
        Matriz_Adyacencia = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()), dtype=int)
        nuevos_individuosNp = np.zeros((mitad_poblacion, numColores, numNodos), dtype=np.int32)
        proba_hijoNp = np.zeros((mitad_poblacion, numColores), dtype=np.float32)
        AMonoNuevoNp = np.zeros((mitad_poblacion), dtype=np.int32)
        TotalHilos = bloques * hilos
        rng_states = create_xoroshiro128p_states((bloques * hilos) * bloques, seed=semilla)
        # ==========================================================================================================
        #=================================== Inicia Algoritmo genético =============================================
        start_time = time.time()
        InicializacionPoblacion(poblacion, probabilidades, G, numNodos)
        if Busqueda_Local(G, AMonocromaticas, poblacion, probabilidades, best, best_ind, termina):
            termina = True
        if not termina: #Continua si NO encontró un individuo con aristas monocromáticas igual a cero
            for gen in range(numGeneraciones):
                for p in range(mitad_poblacion):
                    Padres = EligePadres(poblacion,numNodos,ind_padres,probabilidadeshijo)
                    nuevos_individuos.append(CruzaPadres(Padres,probabilidadeshijo,AMonoNuevo, numNodos, G))
                    ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
                convierte_individuonumpy(nuevos_individuos, probabilidadeshijo, AMonoNuevo, nuevos_individuosNp,proba_hijoNp, AMonoNuevoNp,mitad_poblacion, numColores)

                #Envía al dispositivo CUDA
                d_individuo = cuda.to_device(nuevos_individuosNp)
                d_matriz = cuda.to_device(Matriz_Adyacencia)
                d_prob = cuda.to_device(proba_hijoNp)
                d_AMono = cuda.to_device(AMonoNuevoNp)

                BusquedaLocal_CUDA[bloques, hilos](d_individuo, d_prob, d_AMono, d_matriz, hilos, numNodos, mitad_poblacion, rng_states)  # realiza la búsqueda local en CUDA

                #recibe en CPU
                h_individuos = d_individuo.to_host()
                h_matriz = d_matriz.to_host()
                h_prob = d_prob.to_host()
                h_AMono = d_AMono.to_host()

                for j in range(mitad_poblacion):
                    ActualizaPoblacionCUDA(poblacion, nuevos_individuosNp[j], AMonocromaticas, AMonoNuevoNp[j],probabilidades, proba_hijoNp, ind_padres[j],numNodos)  # reemplaza al hijo con el peor individuo
                    ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
                if Termina_Generacion(bestAnt,AMonoNuevo,gen,NumGenIgual) == 0:
                    break
        imprimeResultados(best,best_ind,start_time,archivo)
        #=================================== Termina Algoritmo genético =============================================

if __name__ == '__main__':
    h=0
    while h < 2:
        probabilidad = 0.4
        while probabilidad <= 0.7:
            numColores = 2
            while numColores < 13:
                #for semilla in range(1,11):
                main()
                numColores=numColores + 2
            probabilidad= probabilidad + 0.1
        #tam_poblacion = tam_poblacion + 128
        h=h+1
        hilos= hilos * 2
        bloques = bloques - 1
    winsound.MessageBeep()
