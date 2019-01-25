import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import random
import numpy as np
from heapq import heappush, heappop, _heapify_max


'''Crea un grafo aleatorio'''
def crea_grafo_aleatorio(numNodos, prob):
    return nx.fast_gnp_random_graph(numNodos,prob) #Genera un grafo aleatorio con un número de nodos
                                                    #y una cierta probabilidad


'''Muestra grafo en una imagen'''
def Muestra_Grafo(G):
    pos = nx.circular_layout(G, scale=5)
    #pos = nx.spring_layout(G, scale=5)  #Genera la posición del grafo
    nx.draw(G, pos, with_labels=True)    #Dibuja el grafo de acuerdo a sus parámetros
    plt.draw()  #Crea el dibujo
    plt.show()  #muestra el dibujo



'''Colorea el grafo con 4 colores: recibe el grafo y las bolsas de colores'''
def colorea_grafo(G, Bolsas_colores):
    color_map = []  #crea un mapa de colores que especifica los colores de cada nodo
    for id in G.nodes:  #Recorre la lista de ids del grafo
        if id in Bolsas_colores[0]: #si el nodo se encuentra en la bolsa 0 lo colorea azul
            color_map.append('blue')
        elif id in Bolsas_colores[1]:
            color_map.append('red') #si el nodo se encuentra en la bolsa 1 lo colorea rojo
        elif id in Bolsas_colores[2]:
            color_map.append('green')   #si el nodo se encuentra en la bolsa 2 lo colorea verde
        elif id in Bolsas_colores[3]:
            color_map.append('purple')   #si el nodo se encuentra en la bolsa 2 lo colorea verde
        else:
            color_map.append('yellow')  #si el nodo no se encuentra en ninguna otra bolsa lo colorea amarillo
    #pos = nx.fruchterman_reingold_layout(G, scale=100)  # Genera la posición del grafo
    pos = nx.circular_layout(G, scale=100)  # Genera la posición del grafo
    nx.draw(G, pos, node_color=color_map, with_labels=True)  #dibuja el grafo con el mapa de colores
    plt.show()  #Muestra el Grafo coloreado



'''Función que se encarga de leer el archivo (benchmark) y crear el grafo'''
def crea_grafo(namefile):
    fh = open(namefile, 'rb')   #abre el archivo
    H = nx.read_edgelist(fh, nodetype=int)  #lee el archivo y toma los datos para
                                            #crear el grafo con datos enteros
    fh.close()  #cierra el archivo
    return H #regresa el grafo



'''Crea el heap con los nodos del mayor grado al menor'''
'''Genera una lista que contiene el id del nodo, su grado y su número de clase'''
def crea_Heap(G, numColores,numNodos):
    heap = []
    clase = numColores * -1 #se multiplica por -1 debido a que con el maxheap queremos en mínimo número de clases
    for nodo in range(1,numNodos+1):
        if nodo in G.nodes:
            grad = G.degree(nodo)
            heappush(heap, (clase,grad,nodo, set()))   #agregamos la tupla(clase,grado,id, conjunto clases válidas)
                                                                     #al heap
        else:
            heappush(heap, (clase, 0 ,nodo, set()))  # agregamos la tupla(clase,grado,id, conjunto clases válidas)
                                                                    # al heap
    heapq._heapify_max(heap)    #hace un max heap
    return heap #regresa el heap



'''Muestra el contenido del heap'''
def ver_heap(heap):
    print('Heap: ')
    while heap: #Mientras el heap No esté vacío saca uno a uno los elementos del tope
        print(heappop(heap))



'''Verifica si el nodo actual se encuentra dentro de la bolsa i'''
def es_compatible_bolsa(i, G, id, Bolsas):
    if bool(Bolsas[i]): #verifica que el diccionario (bolsa i) no esté vacía
        adj = G[id]   #crea la lista con las adyacencias del nodo actual
        for nodo in adj:    #recorre la lista de adyacencias
            if nodo in Bolsas[i]:   #pregunta si el nodo adyacente se encuentra dentro de la bolsa i
                return 1    #si existe una adyacencia en la bolsa, retorna un uno
    return 0    #si NO existe adyacencia, o la bolsa está vacía retorna un cero



'''Disminuye el número de clases de los nodos adyacentes al nodo actual'''
def disminuye_clases(G, heap, id, ind_Bolsa,grado):
    if grado > 0:
        adyacentes = G[id]    # crea una lista con los nodos adyacentes al nodo actual
        for id2 in adyacentes:  # Recorre la lista de nodos adyacentes
            ids = [ids[2] for ids in heap]  # crea una lista con los ids dentro de la tupla(como están acomodados en heap)
            if id2 in ids:  # pregunta si el id de adyacencias se encuentra de la lista de 'ids'
                temp = list(heap[ids.index(id2)]) # guarda la tupa con los valores (del índice donde lo encontró)
                heap[ids.index(id2)] = heap[-1]   # mueve el valor viejo al final del heap
                heap.pop()  # lo elimina
                if not ind_Bolsa in temp[3]:
                    temp[0] = temp[0] - 1 # aumenta en uno la clase
                temp[3].add(ind_Bolsa)
                heappush(heap, tuple(temp))  # ingresa la tupla con los valores actualizados



def cuenta_nodos(individuo, numColores, numNodos):
    probabilidades=[]
    probabilidades = {k:len(v)/numNodos for k, v in individuo.items()}
    return probabilidades


###
def crea_individuo(G, numColores,numNodos):
    heap = [] #Es la lista que contiene el id, grado y clase
    Bolsas_colores = {}   #Es el diccionario que representan las bolsas de colores
    i = 0   #Es el índice de las bolsas de colores

    #print('Grafo: ')
    #print(list(G))  #Muestra la lista de elementos del grafo
    #Muestra_Grafo(G)   #crea una imagen del grafo con un sólo color

    heap = crea_Heap(G, numColores,numNodos) #toma los datos del grafo y regresa la lista
    #ver_heap(heap[:]) #Muestra los elementos del heap (envía copia del heap)
    #print('\n')
    Bolsas_colores = {k: {} for k in range(numColores)} #Crea un diccionario donde cada key es un color y el valor
                                                        #contiene su propio diccionario el cual contendrá los nodos
    #print(Bolsas_colores)
    while heap: #recorre el heap hasta que esté vacío
        clase,grado,id,conj = heappop(heap) #toma los datos del heap (el primer nodo)
        #ver_heap(heap[:])  # Muestra los elementos del heap (envía copia del heap)
        while i< numColores:    #busca la bolsa que no contenga adyacencias
            #if es_compatible_bolsa(i, G, id, Bolsas_colores):
            if i in conj:
                i = i + 1
            else: break
        if i < numColores:   #inserta el nodo actual en la bolsa que NO contiene adyacencias
            Bolsas_colores[i][id] = id
        else:   #El nodo contiene adyacencias en todas las bolsas (lo ingresa en una aleatoria)
            Bolsas_colores[random.randint(0, numColores-1)][id] = id
        disminuye_clases(G, heap, id, i,grado)  # disminuye en uno las clases adyacentes
        #heapq._heapify_max(heap)  # Vuelve a ordenar el heap
        i = 0

    #print('Bolsas de colores: ')
    #print(Bolsas_colores)   #Muestra la bolsa de colores
    #colorea_grafo(G,Bolsas_colores) #Crea una imagen del grafo con 4 colores
    return Bolsas_colores


