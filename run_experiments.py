import os
from itertools import product

jacobian_dropout = [1]
dropout_rates = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]

for jacobian_or_vanilla, dropout_rate in product(jacobian_dropout, dropout_rates):
    command = f"python3 train.py --dropout_rate {dropout_rate} --jacobian_dropout {jacobian_or_vanilla}"
    os.system(command)
