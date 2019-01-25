import networkx as nx
import numba
from numba import cuda, prange
from numba.cuda.random import xoroshiro128p_uniform_float32
import math
import numpy as np
import random


T = 2.7
k = 1.38064852
iteraciones = 20
busqueda_vecindario = 1000


'''Mueve un nodo al azar a otra bolsa (se mueve en el vecindario), solo realiza el movimiento si genera un progreso'''
#@numba.jit
def vecino_escalando(G,individuo, probabilidades,AristasMono, numColores):
    for i in range(busqueda_vecindario):
        while True:
            bolsaProbabilistica = bolsaAleatoriaProbabilidad(probabilidades, numColores) #Elige una bolsa con probabilidad a su número de nodos
            if individuo[bolsaProbabilistica]:  #verifica que la bolsa que eligió no esté vacía
                break
        dic = dict(individuo[bolsaProbabilistica])   #Toma el diccionario con los nodos de la bolsa elegida
        nodo = random.choice(list(dic.keys()))   #Elige un nodo al azar de la bolsa
        monoAct = num_mono_bolsaM(dic, nodo, G)  # calcula el número de aristas monocromáticas del nodo en la bolsa elegida
        BolsaNueva = BolsaAleatoria(bolsaProbabilistica, numColores)
        monopost = num_mono_bolsaM(individuo[BolsaNueva], nodo, G)
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



'''Fórmula de Boltzmann para determinar si acepta o no el cambio de vecindario'''
@numba.jit()
def probabilidadAceptarCUDA(delta, rng_states, id):
    if delta < 0:
        return True
    else:
        P = math.exp(-delta/k*T)
        r = xoroshiro128p_uniform_float32(rng_states, id) # selecciona un número al azar del cero al uno
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



'''Elige una bolsa aleatoria que no sea la que ya eligió probabilísticamente'''
@numba.jit()
def BolsaAleatoriaCUDA(bolsa,numColores,rng_states,id):
    while True:
        rand = int(xoroshiro128p_uniform_float32(rng_states, id)*numColores) #Elige una bolsa aleatoria
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
    return i



'''Elige una bolsa con probabilidad basada en el número de nodos contenidos en la bolsa'''
@numba.jit()
def bolsaAleatoriaProbabilidadCUDA (probabilidades, numColores,rng_states,id):
    r = xoroshiro128p_uniform_float32(rng_states, id) #selecciona un número al azar del cero al uno
    l = 0
    for i in range(numColores): #recorre hasta el número de bolsas
        if (r >= l and r < l + probabilidades[i]):  #si cae entre l y la probabilidad de la bolsa i
            return i    #retorna el indice i
        else:
            l = l + probabilidades[i]    #si no a la variable l le suma probabilidad de la bolsa i
    return i



'''Calcula el número de aristas monocromáticas de la bolsa, dado un nodo, una bolsa y el grafo'''
def num_mono_bolsa( bolsa, nodo, G):
    num_mono = 0    #Empieza con el NAM en cero
    if nodo in G.nodes:
        adj = G[nodo]   #Toma la lista de adyacencia del nodo mandado como parámetro
        for i in bolsa: #Toma el primer nodo en la bolsa
            if i in G.nodes:
                adj2 = G[i] #Toma su lista de adyacencia
                if G.has_edge(nodo, i) or G.has_edge(i, nodo):  #Pregunta si existe adyacencia entre el nodo mandado y el nodo de la bolsa
                    num_mono = num_mono + 1 #Aumenta en uno el NAM
    return num_mono #Regresa el total de NAM



'''Calcula el número de aristas monocromáticas de la bolsa, dado un nodo, una bolsa y la matriz de adyacencia'''
def num_mono_bolsaM( bolsa, nodo, M):
    num_mono = 0    #Empieza con el NAM en cero
    for i in bolsa: #Toma el primer nodo en la bolsa
        if M[nodo-1, i-1] == 1 or M[i-1, nodo-1] == 1:  #Pregunta si existe adyacencia entre el nodo mandado y el nodo de la bolsa
            num_mono = num_mono + 1 #Aumenta en uno el NAM
    return num_mono #Regresa el total de NAM



'''Calcula el número de aristas monocromáticas de la bolsa, dado un nodo, una bolsa y la matriz de adyacencia'''
@numba.jit()
def num_mono_bolsaCUDA( bolsa, nodo, M, numNodos):
    num_mono = 0    #Empieza con el NAM en cero
    for i in range(numNodos): #Toma el primer nodo en la bolsa
        if bolsa[i] == 1:   #Si el nodo se encuentra en la bolsa
            if M[nodo, i] == 1 or M[i, nodo] == 1:  #Pregunta si existe adyacencia entre el nodo mandado y el nodo de la bolsa
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



'''Verifica si la bolsa elegida está vacía o no, toma la bolsa
y recorre la búsqueda sobre todos los nodos, si en todos es cero,
entonces la bolsa está vacía'''
@numba.jit()
def es_vacia(Bolsa,numNodos):
    sum = 0
    for i in range(numNodos):
        sum += Bolsa[i]
    return sum


@numba.jit()
def imprimeIndv(individuo,numNodos,Colores):
    for i in range(Colores):
        for j in range(numNodos):
            print("Individuo ",i,j,":")
            print (individuo[i][j])



bolsaProbabilidad_gpu = cuda.jit(device=True)(bolsaAleatoriaProbabilidadCUDA)
esVacia_gpu = cuda.jit(device=True)(es_vacia)
NAMBolsa_gpu = cuda.jit(device=True)(num_mono_bolsaCUDA)
bolsaAleatoria_gpu = cuda.jit(device=True)(BolsaAleatoriaCUDA)
probAcepta_gpu = cuda.jit(device=True)(probabilidadAceptarCUDA)

@numba.jit()
def Busqueda_MetropolisCUDA(M,individuo,probabilidades,AristMono,numColores,numNodos,rng_states,id):
    if AristMono != 0:
        for i in prange(busqueda_vecindario):
            r = 0
            vacia = 0
            while vacia == 0:
                bolsaProbabilistica = int(bolsaProbabilidad_gpu(probabilidades,numColores,rng_states,id))  # Elige una bolsa con probabilidad a su número de nodos
                vacia = esVacia_gpu(individuo[bolsaProbabilistica],numNodos) #verifica que la bolsa no esté vacía
            while r == 0:    #Selecciona un nodo al azar
                nodo = int(xoroshiro128p_uniform_float32(rng_states, id)*numNodos)  #selecciona un número al azar del cero al número de nodos
                r = individuo[bolsaProbabilistica][nodo]
            monoAct = NAMBolsa_gpu(individuo[bolsaProbabilistica], nodo, M,numNodos)  # calcula el número de aristas monocromáticas del nodo en la bolsa elegida
            BolsaNueva = bolsaAleatoria_gpu(bolsaProbabilistica, numColores,rng_states,id)
            monopost = NAMBolsa_gpu(individuo[BolsaNueva], nodo, M,numNodos)
            delta = monopost - monoAct
            if probAcepta_gpu(delta,rng_states,id):
                AristMono = AristMono + delta
                individuo[bolsaProbabilistica][nodo] = 0  # Elimina el nodo de la bolsa
                individuo[BolsaNueva][nodo] = 1  # Inserta el nodo en otra bolsa al azar
            if AristMono == 0:
                break


@numba.jit()
def Busqueda_EscalandoCUDA(M,individuo,probabilidades,AristMono,numColores,numNodos,rng_states,id):
    if AristMono != 0:
        for i in prange(busqueda_vecindario):
            r = 0
            vacia = 0
            while vacia == 0:
                bolsaProbabilistica = int(bolsaProbabilidad_gpu(probabilidades,numColores,rng_states,id))  # Elige una bolsa con probabilidad a su número de nodos
                vacia = esVacia_gpu(individuo[bolsaProbabilistica],numNodos) #verifica que la bolsa no esté vacía
            while r == 0:    #Selecciona un nodo al azar
                nodo = int(xoroshiro128p_uniform_float32(rng_states, id)*numNodos)  #selecciona un número al azar del cero al número de nodos
                r = individuo[bolsaProbabilistica][nodo]
            monoAct = NAMBolsa_gpu(individuo[bolsaProbabilistica], nodo, M,numNodos)  # calcula el número de aristas monocromáticas del nodo en la bolsa elegida
            BolsaNueva = bolsaAleatoria_gpu(bolsaProbabilistica, numColores,rng_states,id)
            monopost = NAMBolsa_gpu(individuo[BolsaNueva], nodo, M,numNodos)
            delta = monopost - monoAct
            if monopost > monoAct:
                AristMono = AristMono + delta
                individuo[bolsaProbabilistica][nodo] = 0  # Elimina el nodo de la bolsa
                individuo[BolsaNueva][nodo] = 1  # Inserta el nodo en otra bolsa al azar
            if AristMono == 0:
                break



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
