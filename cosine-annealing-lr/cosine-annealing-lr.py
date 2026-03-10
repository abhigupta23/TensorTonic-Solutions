import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """

    rate = 0.0
    

    cosine = math.cos((math.pi * current_step) / total_steps)
    rate = min_lr + ((base_lr-min_lr)*(1 + cosine))/2   

    return rate