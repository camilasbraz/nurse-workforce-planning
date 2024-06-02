import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import os


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
    print(total_horas)
    print(total_horas >= horas_necessarias)
    
    # Critério 1: Cobertura de horas de enfermagem necessárias
    cobertura = np.sum(total_horas >= horas_necessarias)
    
    # Critério 2: Respeito aos descansos mínimos entre turnos
    violacoes_descanso = 0
    for i in range(num_enfermeiros):
        turnos = np.where(individuo[i] == 1)[0]
        if any(np.diff(turnos) == 1):  # Verifica se há turnos consecutivos
            violacoes_descanso += 1
    
    # Critério 3: Distribuição justa dos turnos entre enfermeiros
    turnos_por_enfermeiro = np.sum(individuo, axis=1)
    distribuicao_justa = np.std(turnos_por_enfermeiro)
    
    # Critério 4: Respeito à porcentagem mínima de enfermeiros necessários por tipo de paciente
    violacao_tipo_enfermeiro = 0
    if enfermeiros_por_tipo is None:
        enfermeiros_por_tipo = [0.33, 0.33, 0.34] 
    for tipo, pct_min in enumerate(enfermeiros_por_tipo):
        num_enfermeiros_tipo = np.sum(individuo[:, tipo % num_turnos])
        if num_enfermeiros_tipo < pct_min * num_enfermeiros:
            violacao_tipo_enfermeiro += (pct_min * num_enfermeiros - num_enfermeiros_tipo)
    
    valor_fitness = cobertura - (violacoes_descanso + distribuicao_justa + violacao_tipo_enfermeiro)
    
    return valor_fitness,


def run_genetic_algorithm(params, the, n_enfermeiros, n_turnos):
    # Configuração do algoritmo genético usando DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

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

def main():
    n_turnos = 10
    n_enfermeiros = 5

    # Ajustando o número de genes para ser compatível com o reshape
    n_genes = n_turnos * n_enfermeiros

    p = [40, 12, 5, 2]
    the = the_por_turno(p, n_turnos)

    params_space = [
        {'pop_size': 50, 'n_genes': n_genes, 'cxpb': 0.5, 'mutpb': 0.2, 'n_gen': 20},
        {'pop_size': 100, 'n_genes': n_genes, 'cxpb': 0.7, 'mutpb': 0.1, 'n_gen': 30},
        {'pop_size': 200, 'n_genes': n_genes, 'cxpb': 0.9, 'mutpb': 0.05, 'n_gen': 40}
    ]
    
    results = []

    for params in params_space:
        pop, hof, stats, logbook = run_genetic_algorithm(params, the, n_enfermeiros, n_turnos)
        best_ind = hof[0]
        results.append({
            'params': params,
            'fitness': best_ind.fitness.values[0],
            'logbook': logbook
        })

    # Salvando resultados em CSV
    csv_data = []
    for result in results:
        log_data = {
            "pop_size": result['params']['pop_size'],
            "cxpb": result['params']['cxpb'],
            "mutpb": result['params']['mutpb'],
            "n_gen": result['params']['n_gen'],
            "best_fitness": result['fitness']
        }
        csv_data.append(log_data)
    df = pd.DataFrame(csv_data)
    df.to_csv('results.csv', index=False)
    
    # Plotando e salvando os resultados
    plt.figure(figsize=(10, 5))
    for result in results:
        gen = result['logbook'].select("gen")
        avg_fitness_values = result['logbook'].select("avg")
        plt.plot(gen, avg_fitness_values, label=f"Pop Size: {result['params']['pop_size']}, CxPB: {result['params']['cxpb']}, MutPB: {result['params']['mutpb']}")
    
    plt.xlabel("Generation")
    plt.ylabel("Avg Fitness")
    plt.title("Comparação da Evolução do Fitness Médio")
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_comparison.png')

    for i, result in enumerate(results):
        best_individual = np.array(result['logbook'][-1]["individual"]).reshape(n_enfermeiros, n_turnos)
        df = pd.DataFrame(best_individual, 
                          columns=[f'Turno {j+1}' for j in range(n_turnos)],
                          index=[f'Enfermeiro {j+1}' for j in range(n_enfermeiros)])
        
        # Gráfico de barras da distribuição de turnos por enfermeiro
        turnos_por_enfermeiro = np.sum(best_individual, axis=1)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=[f'Enfermeiro {j+1}' for j in range(n_enfermeiros)], y=turnos_por_enfermeiro)
        plt.xlabel("Enfermeiro")
        plt.ylabel("Número de Turnos")
        plt.title(f"Distribuição de Turnos por Enfermeiro - Config {i+1}")
        plt.savefig(f'turnos_por_enfermeiro_config_{i+1}.png')

        # Heatmap da escala de turnos
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="YlGnBu", cbar=False, linewidths=0.5, annot=True)
        plt.xlabel("Turnos")
        plt.ylabel("Enfermeiros")
        plt.title(f"Heatmap da Escala de Turnos - Config {i+1}")
        plt.savefig(f'heatmap_escala_turnos_config_{i+1}.png')

if __name__ == "__main__":
    main()
