import networkx as nx
import numba
import math
import numpy as np
import random


T = 2.7
k = 1.38064852
iteraciones = 20
busqueda_vecindario = 1000


'''Mueve un nodo al azar a otra bolsa (se mueve en el vecindario), solo realiza el movimiento si genera un progreso'''
def vecino_escalando(G,individuo, probabilidades,AristasMono, numColores):
    for i in range(busqueda_vecindario):
        while True:
            bolsaProbabilistica = bolsaAleatoriaProbabilidad(probabilidades, numColores) #Elige una bolsa con probabilidad a su número de nodos
            if individuo[bolsaProbabilistica]:  #verifica que la bolsa que eligió no esté vacía
                break
        dic = dict(individuo[bolsaProbabilistica])   #Toma el diccionario con los nodos de la bolsa elegida
        nodo = random.choice(list(dic.keys()))   #Elige un nodo al azar de la bolsa
        monoAct = num_mono_bolsa(dic, nodo, G)  # calcula el número de aristas monocromáticas del nodo en la bolsa elegida
        BolsaNueva = BolsaAleatoria(bolsaProbabilistica, numColores)
        monopost = num_mono_bolsa(individuo[BolsaNueva], nodo, G)
        if monopost != 0 and monopost < monoAct:
            delta = monopost - monoAct
            AristasMono = AristasMono + delta
            del individuo[bolsaProbabilistica][nodo]  # Elimina el nodo de la bolsa
            individuo[BolsaNueva][nodo] = nodo  # Inserta el nodo en otra bolsa al azar
        if AristasMono == 0:
            break
    return AristasMono



'''Mueve un nodo al azar a otra bolsa (se mueve en el vecindario), no desecha malos movimientos si no que 
elige en base a una probabilidad si realiza o no el movimiento'''
def vecino_metropolis(G,individuo, probabilidades,AristasMono, numColores):
    for i in range(busqueda_vecindario):
        while True:
            bolsaProbabilistica = bolsaAleatoriaProbabilidad(probabilidades, numColores) #Elige una bolsa con probabilidad a su número de nodos
            if individuo[bolsaProbabilistica]:  #verifica que la bolsa que eligió no esté vacía
                break
        dic = dict(individuo[bolsaProbabilistica])   #Toma el diccionario con los nodos de la bolsa elegida
        nodo = random.choice(list(dic.keys()))   #Elige un nodo al azar de la bolsa
        monoAct = num_mono_bolsa(dic, nodo, G) #calcula el número de aristas monocromáticas del nodo en la bolsa elegida
        BolsaNueva = BolsaAleatoria(bolsaProbabilistica, numColores)
        monopost = num_mono_bolsa(individuo[BolsaNueva], nodo, G)
        delta = monopost - monoAct
        if probabilidadAceptar(delta):
            AristasMono = AristasMono + delta
            del individuo[bolsaProbabilistica][nodo]  # Elimina el nodo de la bolsa
            individuo[BolsaNueva][nodo] = nodo  # Inserta el nodo en otra bolsa al azar
        if AristasMono == 0:
            break
    return AristasMono



'''Fórmula de Boltzmann para determinar si acepta o no el cambio de vecindario'''
def probabilidadAceptar(delta):
    if delta < 0:
        return True
    else:
        P = math.exp(-delta/k*T)
        r = random.uniform(0, 1)  # selecciona un número al azar del cero al uno
        if r < P:
            return True
        else:
            return False



'''Elige una bolsa aleatoria que no sea la que ya eligió probabilísticamente'''
def BolsaAleatoria(bolsa,numColores):
    while True:
        rand = random.randint(0, numColores - 1)    #Elige una bolsa aleatoria
        if rand != bolsa:   #si la bolsa es diferente a la elegida
            return rand #Regresa el indice de la bolsa



'''Elige una bolsa con probabilidad basada en el número de nodos contenidos en la bolsa'''
def bolsaAleatoriaProbabilidad (probabilidades, numColores):
    r = random.uniform(0, 1) #selecciona un número al azar del cero al uno
    l = 0
    for i in range(numColores): #recorre hasta el número de bolsas
        if (r >= l and r < l + probabilidades[i]):  #si cae entre l y la probabilidad de la bolsa i
            return i    #retorna el indice i
        else:
            l = l + probabilidades[i]    #si no a la variable l le suma probabilidad de la bolsa i



'''Calcula el número de aristas monocromáticas de la bolsa, dado un nodo, una bolsa y el grafo'''
def num_mono_bolsa( bolsa, nodo, G):
    num_mono = 0    #Empieza con el NAM en cero
    if nodo in G.nodes:
        #adj = G[nodo]   #Toma la lista de adyacencia del nodo mandado como parámetro
        for i in bolsa: #Toma el primer nodo en la bolsa
            if i in G.nodes:
                #adj2 = G[i] #Toma su lista de adyacencia
                if G.has_edge(nodo, i) or G.has_edge(i, nodo):  #Pregunta si existe adyacencia entre el nodo mandado y el nodo de la bolsa
                    num_mono = num_mono + 1 #Aumenta en uno el NAM
    return num_mono #Regresa el total de NAM



'''Calcula el número monocromático del individuo'''
def numeroAristasMono(G, individuo, numColores):
    num_mono = 0
    for i in range(numColores): #Toma las bolsas del individuo
        nodos_bolsa = list(individuo[i])    #Toma los nodos contenidos en la bolsa actual
        for j in range(len(nodos_bolsa)-1): #Toma en orden los nodos
            r = j+1
            while r < len(nodos_bolsa): #Toma el nodo siguiente al actual
                if G.has_edge(nodos_bolsa[j], nodos_bolsa[r]) or G.has_edge(nodos_bolsa[r], nodos_bolsa[j]): #Pregunta si hay una
                    num_mono = num_mono + 1 #suma en uno las aristas monocromáticas                          #adyacencia de ambas formas
                r = r+1 #Toma el nodo que sigue
    return num_mono #regresa el total de NAM



'''Recibe el grafo para obtener las adyacencias de los nodos, el individuo que contiene la información, las probabilidades
que contiene la probabilidad de las bolsas de acuerdo al número de nodos contenidos, Las aristas que contiene el número de
aristas monocromáticas del individuo, y el número de colores para recorrer las bolsas'''
def Busqueda_Metropolis(G, individuo, probabilidades,Aristas, numColores):
    if Aristas != 0:    #Si el último evaluado es diferente de cero continua con la búsqueda local
        AristasDesp = vecino_metropolis(G, individuo, probabilidades,Aristas, numColores) #Realiza la búsqueda Metrópolis
        return individuo, AristasDesp
    else:
        return individuo, Aristas



'''Recibe el grafo para obtener las adyacencias de los nodos, el individuo que contiene la información, las probabilidades
que contiene la probabilidad de las bolsas de acuerdo al número de nodos contenidos, Las aristas que contiene el número de
aristas monocromáticas del individuo, y el número de colores para recorrer las bolsas'''
def Busqueda_Escalando(G, individuo, probabilidades, Aristas, numColores):
    if Aristas != 0:  # Si el último evaluado es diferente de cero continua con la búsqueda local
        AristasDesp = vecino_escalando(G, individuo, probabilidades,Aristas, numColores)
        return individuo, AristasDesp
    else:
        return individuo, Aristas
