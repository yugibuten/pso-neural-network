"""Unit tests for activations, ANN, data, metrics, coupling and PSO."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.activations import logistic, relu, tanh
from src.ann import ANN
from src.data import prepare_data, load_data
from src.metrics import classification_metrics, confusion_matrix
from src.coupling import make_fitness, apply_particle, particle_dim
from src.pso import PSO
from experiments.configs import BASE, architecture_configs, budget_configs
from experiments.runner import select_training


class CoreTests(unittest.TestCase):
    def test_activations(self):
        np.testing.assert_allclose(logistic([-1000,0,1000]), [0,.5,1])
        np.testing.assert_equal(relu([-1,0,2]), [0,0,2])
        np.testing.assert_allclose(tanh([0]), [0])

    def test_hand_weights_and_roundtrip(self):
        net=ANN([2,1],['logistic'])
        net.set_params(np.array([1.,-1.,.5]))
        np.testing.assert_allclose(net.forward([[2,1]]), [[logistic(1.5)]])
        other=ANN([2,1],['logistic'])
        other.set_params(net.get_parameters_vector())
        np.testing.assert_equal(net.forward([[2,1]]), other.forward([[2,1]]))
        vector=net.get_parameters_vector()
        net.set_params(vector)
        vector[:]=100
        np.testing.assert_equal(net.get_parameters_vector(),[1,-1,.5])
        with self.assertRaises(ValueError): net.set_params(np.zeros(2))

    def test_deep_shape(self):
        net=ANN([4]+[16]*8+[1],['relu']*8+['logistic'])
        self.assertEqual(net.n_params,80+7*272+17)
        self.assertEqual(net.forward(np.zeros((5,4))).shape,(5,1))
        self.assertEqual(ANN([4,8,1],['tanh','logistic']).n_params,49)

    def test_data(self):
        train,test,y,z,scaler=prepare_data()
        self.assertEqual(train.shape,(960,4))
        self.assertEqual(test.shape,(412,4))
        np.testing.assert_allclose(train.mean(axis=0),0,atol=1e-12)
        X,_=load_data()
        self.assertFalse(np.allclose(scaler.mean_,X.mean(axis=0)))
        np.testing.assert_allclose(train,prepare_data()[0])

    def test_metrics(self):
        y,p=[0,0,1,1],[0,1,0,1]
        np.testing.assert_equal(confusion_matrix(y,p),[[1,1],[1,1]])
        self.assertEqual(classification_metrics(y,p)['f1'],.5)
        self.assertEqual(classification_metrics([0],[0])['f1'],0)

    def test_fitness_and_encoding(self):
        net=ANN([2,2,1],['tanh','logistic'])
        vector=np.zeros(net.n_params)
        X=np.zeros((2,2)); y=np.array([0,1])
        self.assertEqual(make_fitness(net,X,y)(vector),.5)
        self.assertEqual(make_fitness(net,X,y,'mse')(vector),-.25)
        self.assertAlmostEqual(make_fitness(net,X,y,'bce')(vector),-np.log(2))
        encoded=np.r_[vector,2.5]
        self.assertEqual(particle_dim(net,True),len(encoded))
        apply_particle(net,encoded,True)
        self.assertEqual(net.activations,['relu','logistic'])

    def test_sphere_and_reproducibility(self):
        solver=PSO(3,lambda x:-float(x@x),swarm_size=30,bounds=(-5,5),seed=0)
        a,score,history=solver.run(200)
        self.assertLess(-score,1e-5)
        self.assertEqual(solver.n_evaluations,6000)
        self.assertTrue(np.all(np.diff(history)>=0))
        b,score2,history2=solver.run(200)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(history,history2)

    def test_informant_term_is_used(self):
        # With only gamma nonzero, self-only informants cannot move particles;
        # the full informant topology must move at least one non-best particle.
        kw=dict(dim=2,fitness_fn=lambda x:-float(x@x),swarm_size=5,
                alpha=0,beta=0,gamma=1,delta=0,seed=4)
        self_only=PSO(**kw,n_informants=1); self_only.run(2)
        initial=PSO(**kw,n_informants=1); initial.run(1)
        full=PSO(**kw,n_informants=5); full.run(2)
        np.testing.assert_array_equal(initial.positions,self_only.positions)
        self.assertFalse(np.array_equal(initial.positions,full.positions))

    def test_boundaries_and_budget(self):
        for boundary in ('clip','reflect','reset'):
            seen=[]
            def objective(x):
                self.assertTrue(np.all((x>=-1)&(x<=1)))
                seen.append(x)
                return -float(x@x)
            solver=PSO(2,objective,swarm_size=5,bounds=(-1,1),epsilon=100,boundary=boundary,seed=2)
            solver.run(5)
            self.assertEqual(len(seen),25)
        for config in budget_configs(BASE):
            self.assertIn(config['swarm_size']*config['iterations'],[500,2000])
        self.assertEqual(len(architecture_configs()),60)

    def test_selection_ignores_test_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            rows=[dict(name=n,seed=s,train_accuracy=t,test_accuracy=1-t,config_json=json.dumps(dict(BASE,name=n)))
                  for n,t in [('a',.8),('b',.9)] for s in range(10)]
            pd.DataFrame(rows).to_csv(path/'input.csv',index=False)
            (path/'input.metadata.json').write_text(json.dumps(dict(
                seeds=list(range(10)), configs=[dict(BASE,name=n) for n in ('a','b')])))
            select_training(path/'input.csv',path/'chosen.json')
            self.assertEqual(json.loads((path/'chosen.json').read_text())['name'],'b')


if __name__ == '__main__': unittest.main()
