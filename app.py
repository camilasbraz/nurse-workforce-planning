import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms


def init_individual(N, T):
    return [random.randint(0, 1) for _ in range(N * T)]

def init_population(n, N, T):
    return [creator.Individual(init_individual(N, T)) for _ in range(n)]

def the_por_turno(pacientes, T):
    ht = [2, 3, 5, 5]  # Necessidade de horas de enfermagem por paciente por turno
    the = np.array(pacientes) * np.array(ht)
    the_repeated = np.repeat(the, T // len(the))
    if len(the_repeated) < T:
        the_repeated = np.append(the_repeated, the[:T - len(the_repeated)])
    return the_repeated

def fitness(individuo, horas_necessarias, num_enfermeiros, num_turnos, enfermeiros_por_tipo=None):
    individuo = np.array(individuo).reshape(num_enfermeiros, num_turnos)
    total_horas = np.sum(individuo, axis=0) * 12  # Supondo turnos de 12 horas
    
    # Critério 1: Cobertura de horas de enfermagem necessárias
    penalidade_cobertura = np.sum(np.abs(total_horas - horas_necessarias))
    
    # Critério 2: Respeito aos descansos mínimos entre turnos
    violacoes_descanso = 0
    for i in range(num_enfermeiros):
        turnos = np.where(individuo[i] == 1)[0]
        if any(np.diff(turnos) == 1):  # Verifica se há turnos consecutivos
            violacoes_descanso += 1
    
    # # Critério 3: Distribuição justa dos turnos entre enfermeiros
    # turnos_por_enfermeiro = np.sum(individuo, axis=1)
    # distribuicao_justa = np.std(turnos_por_enfermeiro)
    
    # # Critério 4: Respeito à porcentagem mínima de enfermeiros necessários por tipo de paciente
    # violacao_tipo_enfermeiro = 0
    # if enfermeiros_por_tipo is None:
    #     enfermeiros_por_tipo = [0.33, 0.33, 0.34]  # Exemplo de porcentagens mínimas necessárias
    # for tipo, pct_min in enumerate(enfermeiros_por_tipo):
    #     num_enfermeiros_tipo = np.sum(individuo[:, tipo % num_turnos])
    #     if num_enfermeiros_tipo < pct_min * num_enfermeiros:
    #         violacao_tipo_enfermeiro += (pct_min * num_enfermeiros - num_enfermeiros_tipo)
    
    # Penalidade por turnos não cobertos
    penalidade_turnos_nao_cobertos = np.sum(total_horas < horas_necessarias)
    
    # Penalidade por turnos consecutivos
    penalidade_turnos_consecutivos = violacoes_descanso
    
    # Combinação dos critérios em um valor de fitness
    valor_fitness = (penalidade_cobertura  * 100) - (penalidade_turnos_nao_cobertos * 100) - (penalidade_turnos_consecutivos * 100)
    # - (distribuicao_justa * 10) - (violacao_tipo_enfermeiro * 10)
    
    return valor_fitness,



def run_genetic_algorithm(params, the, n_enfermeiros, n_turnos):
    # Configuração do algoritmo genético usando DEAP
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=params['n_genes'])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def custom_fitness(individual):
        return fitness(individual, the, n_enfermeiros, n_turnos)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", custom_fitness)

    pop = toolbox.population(n=params['pop_size'])
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

n_turnos = st.number_input('Número de turnos', min_value=1, value=10)
n_enfermeiros = st.number_input('Número de enfermeiros', min_value=1, value=5)

# Ajustando o número de genes para ser compatível com o reshape
n_genes = n_turnos * n_enfermeiros

pop_size = st.number_input('Tamanho da população', min_value=1, value=50)
cxpb = st.number_input('Probabilidade de crossover', min_value=0.0, max_value=1.0, value=0.5)
mutpb = st.number_input('Probabilidade de mutação', min_value=0.0, max_value=1.0, value=0.2)
n_gen = st.number_input('Número de gerações', min_value=1, value=20)

p_input = st.text_input('Pacientes de cada tipo (ex: 40, 12, 5, 2)', '5, 5, 5, 5')
p = list(map(int, p_input.split(',')))

if st.button('Otimizar'):
    the = the_por_turno(p, n_turnos)
    params = {
        'pop_size': pop_size,
        'n_genes': n_genes,
        'cxpb': cxpb,
        'mutpb': mutpb,
        'n_gen': n_gen
    }
    
    pop, hof, stats, logbook = run_genetic_algorithm(params, the, n_enfermeiros, n_turnos)
    best_ind = hof[0]
    # st.write('Melhor solução:', best_ind)
    st.write('Fitness da melhor solução:', best_ind.fitness.values[0])
    

    best_individual = np.array(best_ind).reshape(n_enfermeiros, n_turnos)
    df = pd.DataFrame(best_individual, 
                        columns=[f'Turno {i+1}' for i in range(n_turnos)],
                        index=[f'Enfermeiro {i+1}' for i in range(n_enfermeiros)])
    st.dataframe(df)

    # Gráfico da evolução do fitness ao longo das gerações
    gen = logbook.select("gen")
    min_fitness_values = logbook.select("max")
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

    # Gráfico de barras da distribuição de turnos por enfermeiro
    turnos_por_enfermeiro = np.sum(best_individual, axis=1)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=[f'Enfermeiro {i+1}' for i in range(n_enfermeiros)], y=turnos_por_enfermeiro)
    plt.xlabel("Enfermeiro")
    plt.ylabel("Número de Turnos")
    plt.title("Distribuição de Turnos por Enfermeiro")
    st.pyplot(plt)

    # Heatmap da escala de turnos
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="YlGnBu", cbar=False, linewidths=0.5, annot=True)
    plt.xlabel("Turnos")
    plt.ylabel("Enfermeiros")
    plt.title("Heatmap da Escala de Turnos")
    st.pyplot(plt)
        

