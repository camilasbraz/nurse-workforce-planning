import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms

TAMANHO_GENE = 4

def init_individual(N, T):
    return [random.randint(1, 50) for _ in range(N * T)]

def init_population(n, N, T):
    return [creator.Individual(init_individual(N, T)) for _ in range(n)]

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
    horas_semana = np.dot(horas_pd, pacientes)*7*ist

    porcentagens = [0.33, 0.33, 0.36, 0.42, 0.52]
    porc_enf = porcentagens[np.argmax(pacientes)]
    return horas_semana, porc_enf 

def fitness_funcionarios(individuo, horas_necessarias, porcentagem_enf):
    ''''
    Um indivíduo tem estrutura: [n_enf8h, n_enf12h, n_tec8h, n_tec12h]
    
    Critérios
    1. Somatório de horas é próximo das horas mínimas por semana
    2. Aproximadamente x% das horas são de enfermeiros, x é definido pelo max(PACIENTES_POR_TIPO)
    3. Aproximadamente 2/7 das horas totais são de funcionários de 12h, porque funcionários de 12h não trabalham no fim de semana
    '''
    # Cálculos gerais
    cargas_horarias = np.array([44, 40, 44, 40])
    horas_por_func = individuo * cargas_horarias
    horas_totais = np.sum(horas_por_func)

    # Critério 1: Número de horas é próximo das horas necessárias
    penalidade_1 = np.abs(horas_necessarias - horas_totais)

    # Critério 2: Porcentagem de horas de enfermeiros é próxima do necessário
    penalidade_2 = np.abs(np.sum(horas_por_func[0:2]) - porcentagem_enf*horas_totais)
    
    # Critério 3: Mínimo de funcionários que trabalham 12h
    penalidade_3 = np.abs((horas_por_func[1] + horas_por_func[3]) - 0.29 * horas_totais)
    
    # Combinação dos critérios em um valor de fitness
    valor_fitness = penalidade_1 + penalidade_2 + penalidade_3 

    return valor_fitness, 

def run_genetic_algorithm(params, horas_necessarias, porcentagem_enf):
    # Configuração do algoritmo genético usando DEAP
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))             # define que a fitness é de minimização  
    creator.create('Individual', list, fitness=creator.FitnessMin)          # define a estrutura de um individuo

    toolbox = base.Toolbox()
    max_attr = np.ceil(horas_necessarias / 40)     
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

    for gen in range(params['n_gen']):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=params['cxpb'], mutpb=params['mutpb'])
        fits = map(toolbox.evaluate, offspring)
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        pop = toolbox.select(offspring, k=len(pop))
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        hof.update(pop)

    return pop, hof, stats, logbook

# Streamlit App
st.title('Genetic Algorithm for Nurse Scheduling Optimization')

# Entradas
pop_size = st.number_input('Tamanho da população', min_value=1, value=50)
cxpb = st.number_input('Probabilidade de crossover', min_value=0.0, max_value=1.0, value=0.5)
mutpb = st.number_input('Probabilidade de mutação', min_value=0.0, max_value=1.0, value=0.2)
n_gen = st.number_input('Número de gerações', min_value=1, value=20)

p_input = st.text_input('Pacientes de cada tipo (ex: 40, 12, 5, 2, 1)', '5, 5, 5, 5, 0')
p = list(map(int, p_input.split(',')))
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
    print    
    st.write('Enfermeiros 8h: ', best_individual[0], ' - ',best_individual[0]*44, 'h')
    st.write('Enfermeiros 12h: ', best_individual[1], ' - ',best_individual[1]*40, 'h')    
    st.write('Técnicos 8h: ', best_individual[2], ' - ',best_individual[2]*44, 'h')    
    st.write('Técnicos 12h: ', best_individual[3], ' - ',best_individual[3]*40, 'h')
    st.write('Horas totais: ', np.sum(best_individual * [40,44,40,44]))
    st.write('Horas necessárias: ', the)
    
    # df = pd.DataFrame(best_individual, 
    #                     columns=[f'Turno {i+1}' for i in range(n_turnos)],
    #                     index=[f'Enfermeiro {i+1}' for i in range(n_enfermeiros)])
    # st.dataframe(df)

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
        

