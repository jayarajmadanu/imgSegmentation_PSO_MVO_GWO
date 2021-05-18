import numpy as np
import random


# Initilize search agent positions
def initializationGWO(SearchAgents_no, dim, ub, lb):
    positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        positions[:, i] = random.sample(range(lb, ub), SearchAgents_no)
    return positions


def GWO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj, H):
    num_generations = Max_iteration
    generation_entropy = np.zeros(Max_iteration)

    # initialize alpha pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('-inf')

    # initialize beta pos
    Beta_pos = np.zeros(dim)
    Beta_score = float('-inf')

    # initialize delta pos
    Delta_pos = np.zeros(dim)
    Delta_score = float('-inf')

    Positions = initializationGWO(SearchAgents_no, dim, ub, lb)

    for m in range(Max_iteration):
        for k in range(SearchAgents_no):
            # find fitness score for each wolf
            fitness = fobj(Positions[k, :], H)

            # if fitness score is greater than alpha score, update alpha score
            if fitness > Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[k, :].copy()

            # if fitness score is greater than beta score, update beta score
            if fitness < Alpha_score and fitness > Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[k, :].copy()

            # if fitness score is greater than delta score, update delta score
            if fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[k, :].copy()

        # a decresing value from 2 to 0 for convergence
        a = 2 - m * (2 / Max_iteration)

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()

                # A is the convergence factor
                A1 = 2 * a * r1 - a

                #  C is the oscillating factor
                C1 = 2 * r2

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = abs(Alpha_pos[j] - A1 * D_alpha)

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = abs(Beta_pos[j] - A2 * D_beta)

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = abs(Delta_pos[j] - A3 * D_delta)

                Positions[i, j] = (X1 + X2 + X3) / 3

        generation_entropy[m] = Alpha_score

        if m > 30 and np.std(generation_entropy[m - 30:m]) < 0.01:
            num_generations = m
            break

    res = [Alpha_pos, Alpha_score, num_generations, generation_entropy]
    return res
