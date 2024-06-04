import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms

TAMANHO_GENE = 4
NUM_DIAS = 14
NUM_FDS = 4
# def init_individual(N, T):
#     return [random.randint(1, 50) for _ in range(N * T)]

# def init_population(n, N, T):
#     return [creator.Individual(init_individual(N, T)) for _ in range(n)]

def campos_pacientes():
    cols = st.columns(5)

    # Entradas em cada coluna
    with cols[0]:
        p1 = st.number_input('PCM', min_value=0, value=20, step=1)
    with cols[1]:
        p2 = st.number_input('PCI', min_value=0, value=11, step=1)
    with cols[2]:
        p3 = st.number_input('PCAD', min_value=0, value=5, step=1)
    with cols[3]:
        p4 = st.number_input('PCSI', min_value=0, value=1, step=1)
    with cols[4]:
        p5 = st.number_input('PCIt', min_value=0, value=0, step=1)

    # Lista de pacientes
    p = [p1, p2, p3, p4, p5]
    return p

def the_semana(pacientes, ist):
    ''' Explicação:
        Calcula a quantidade de horas de enfermagem necessária em uma semana de acordo com os tipos de pacientes
        Recebe um vetor de quantidade de pacientes por tipo e um índice de segurança (ist)
                    [PCM    PCI     PCAD    PCSI    PCIt]
        pacientes = [40     12      5       2       0   ]
        horas_pd  = [4      6       10      10      18  ]       
        porc_enf  = [0,33   0,33    0,36    0,42    0,52]

        - O numero de horas totais de enfermagem por semana é:
            sum(pacientes * horas_pd)*7*IST
        - A porcentagem de enfermeiros nessas horas é
            porc_enf[argmax(pacientes)]
    ''' 
    horas_pd = [4, 6, 10, 10, 18]
    horas_semana = np.dot(horas_pd, pacientes)*NUM_DIAS*ist

    porcentagens = [0.33, 0.33, 0.36, 0.42, 0.52]
    porc_enf = porcentagens[np.argmax(pacientes)]
    return horas_semana, porc_enf 

def fitness_funcionarios(individuo, horas_necessarias, porcentagem_enf):
    ''''
    Um indivíduo tem estrutura: [EN_12, EN_9, TE_12, TE_9]
    
    Critérios
    1. Somatório de horas é próximo das horas mínimas por semana
    2. Aproximadamente x% dos funcionários são enfermeiros, x é definido pelo max(PACIENTES_POR_TIPO)
    '''

    # Constantes

    PESO_1 = 100
    PESO_2 = 1
    PESO_3 = 5

    # Num de funcionarios
    en_12, en_9, te_12, te_9 = individuo
    N_en = en_12 + en_9 # total enf
    N_te = te_12 + te_9 # total tec
    N_total = N_en + N_te

    # Critério 1: proporção de nfermeiros e técnicos
    # Proporcoes
    prop_en = N_en / N_total
    prop_te = N_te / N_total

    # Calculando a diferença absoluta
    diff_en = abs(prop_en - porcentagem_enf)
    diff_te = abs(prop_te - (1-porcentagem_enf))

    # Calculando a penalidade de proporção
    penalidade_proporcao = PESO_1 * (diff_en + diff_te)

    # Critério 2: horas mínimas necessárias
    horas_por_dia = (en_12 * 12) + (en_9 * 9) + (te_12 * 12) + (te_9 * 9)
    total_horas = horas_por_dia * NUM_DIAS
    if total_horas < horas_necessarias :
        penalidade_horas = 1000
    else:
        penalidade_horas = PESO_2 * (total_horas - horas_necessarias)

    valor_fitness = penalidade_proporcao + penalidade_horas

    return valor_fitness, 

def run_genetic_algorithm(params, horas_necessarias, porcentagem_enf):
    # Configuração do algoritmo genético usando DEAP
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))             # define que a fitness é de minimização  
    creator.create('Individual', list, fitness=creator.FitnessMin)          # define a estrutura de um individuo

    toolbox = base.Toolbox()
    max_attr = np.ceil(horas_necessarias / 84)     
    toolbox.register('genes', np.random.randint, 0, max_attr + 1)    # um indivíduo so admite valores 0 a max_attr
    toolbox.register('individuo', tools.initRepeat, creator.Individual, toolbox.genes, TAMANHO_GENE)   # um individuo é uma lista de 4 attr_funcionario 
    toolbox.register('populacao', tools.initRepeat, list, toolbox.individuo)    # uma populacao é uma lista de individuos

    def custom_fitness(individual):
        return fitness_funcionarios(individual, horas_necessarias, porcentagem_enf)

    toolbox.register("mate", tools.cxTwoPoint)  # crossover
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=max_attr, indpb=params['mutpb'])   # mutacao (num aleatorio de 1 a max_attr)
    toolbox.register("select", tools.selTournament, tournsize=3)    # selecao de pais por torneio
    toolbox.register("evaluate", custom_fitness)

    pop = toolbox.populacao(n=params['pop_size'])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Critérios de parada
    fitness_threshold = 50  # Critério de parada baseado na aptidão mínima
    best_fitness = float('inf')

    for gen in range(params['n_gen']):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=params['cxpb'], mutpb=params['mutpb'])
        fits = map(toolbox.evaluate, offspring)
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        pop = toolbox.select(offspring, k=len(pop))
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        hof.update(pop)

        current_best_fitness = record['min']
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness

    return pop, hof, stats, logbook

# Streamlit App
st.title('Genetic Algorithm for Nurse Scheduling Optimization')

# Entradas
pop_size = st.number_input('Tamanho da população', min_value=1, value=50)
cxpb = st.number_input('Probabilidade de crossover', min_value=0.0, max_value=1.0, value=0.8)
mutpb = st.number_input('Probabilidade de mutação', min_value=0.0, max_value=1.0, value=0.1)
n_gen = st.number_input('Número de gerações', min_value=1, value=100)

p = campos_pacientes()
ist = st.number_input('Índice de segurança técnica', min_value=1.15, value=1.15)

if st.button('Otimizar'):
    the, porc_enf = the_semana(p, ist)      # Cálculo inicial
    params = {
        'pop_size': pop_size,
        'cxpb': cxpb,
        'mutpb': mutpb,
        'n_gen': n_gen
    }
    pop, hof, stats, logbook = run_genetic_algorithm(params, the, porc_enf)
    best_ind = hof[0]
    # st.write('Melhor solução:', best_ind)
    st.write('Fitness da melhor solução:', best_ind.fitness.values[0])
    
    best_individual = np.array(best_ind)
    en_12, en_9, te_12, te_9 = best_individual
    horas_por_dia = (en_12 * 12) + (en_9 * 9) + (te_12 * 12) + (te_9 * 9)
    total_hours = horas_por_dia * NUM_DIAS
    # Criar um DataFrame para exibir os resultados em uma tabela
    data = {
        'Categoria': ['Enfermeiros 12h', 'Enfermeiros 9h', 'Técnicos 12h', 'Técnicos 9h'],
        'Quantidade': [en_12, en_9, te_12, te_9],
        'Horas por quinzena': horas_por_dia * NUM_DIAS
    }

    df = pd.DataFrame(data)

    # Exibir a tabela
    st.table(df)

    # Exibir o total de horas
    st.write('Horas totais: ', total_hours)
    st.write('Horas necessárias: ', the)
    st.write('Diferença: ', total_hours - the)

    # Exibir porcentagem de enfermeiros
    st.write('"%" enfermeiros: ', ((en_12 + en_9)/(en_12 + en_9 + te_12+te_9))*100, '%')
    st.write('"%" necessária: ', porc_enf)

    # Gráfico da evolução do fitness ao longo das gerações
    gen = logbook.select("gen")
    min_fitness_values = logbook.select("min")
    avg_fitness_values = logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, min_fitness_values, label="Min Fitness")
    plt.plot(gen, avg_fitness_values, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolução do Fitness ao Longo das Gerações")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # # Gráfico de barras da distribuição de turnos por enfermeiro
    # turnos_por_enfermeiro = np.sum(best_individual, axis=1)

    # plt.figure(figsize=(10, 5))
    # sns.barplot(x=[f'Enfermeiro {i+1}' for i in range(n_enfermeiros)], y=turnos_por_enfermeiro)
    # plt.xlabel("Enfermeiro")
    # plt.ylabel("Número de Turnos")
    # plt.title("Distribuição de Turnos por Enfermeiro")
    # st.pyplot(plt)

    # # Heatmap da escala de turnos
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df, cmap="YlGnBu", cbar=False, linewidths=0.5, annot=True)
    # plt.xlabel("Turnos")
    # plt.ylabel("Enfermeiros")
    # plt.title("Heatmap da Escala de Turnos")
    # st.pyplot(plt)
        

