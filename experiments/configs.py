"""Experiment settings. Custom architectures are optional additions to the grid."""
from copy import deepcopy

BASE = dict(layer_sizes=[4, 8, 1], activations=['tanh', 'logistic'],
            swarm_size=20, iterations=100, alpha=.7298, beta=1.496, gamma=1.496,
            delta=0., epsilon=1., n_informants=3, bounds=[-3., 3.],
            boundary='clip', alpha_end=None, fitness='accuracy', encode_activations=False)
CUSTOM_ARCHITECTURES = {}  # Add e.g. 'A': ([4, 8, 1], ['tanh', 'logistic']).


def setting(name, base=None, **changes):
    config = deepcopy(BASE if base is None else base)
    config.update(changes)
    config['name'] = name
    return config


def architecture_configs():
    configs = [setting(f'd{d}_w{w}_{a}', layer_sizes=[4]+[w]*d+[1],
                       activations=[a]*d+['logistic'])
               for d in (1, 2, 3, 5, 8) for w in (2, 4, 8, 16)
               for a in ('tanh', 'logistic', 'relu')]
    configs += [setting(name, layer_sizes=sizes, activations=acts)
                for name, (sizes, acts) in CUSTOM_ARCHITECTURES.items()]
    return configs


def budget_configs(base):
    return [setting(f'budget{swarm*iterations}_{swarm}x{iterations}', base,
                    swarm_size=swarm, iterations=iterations)
            for swarm, iterations in ((10,50),(20,25),(25,20),(50,10),
                                      (10,200),(20,100),(25,80),(50,40))]


def coefficient_configs(base):
    # Each group varies only its named factor (or the beta/gamma balance).
    configs = [setting('balance_'+name, base, beta=b, gamma=g)
               for name,b,g in [('balanced',1.496,1.496),('cognitive',2.244,.748),('social',.748,2.244)]]
    configs += [setting(f'alpha_{a}', base, alpha=a) for a in (.4,.7298,.9)]
    configs += [setting(f'delta_{d}', base, delta=d) for d in (0.,.5,1.)]
    return configs


def extension_configs(base):
    configs = [setting('boundary_'+b, base, boundary=b) for b in ('clip','reflect','reset')]
    configs += [setting('fitness_'+f, base, fitness=f) for f in ('accuracy','mse','bce')]
    configs += [setting('activation_'+str(e), base, encode_activations=e) for e in (False,True)]
    return configs
