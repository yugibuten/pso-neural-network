"""Metadata, extension mechanics and synthetic artifact reconstruction checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.ann import ANN
from src.coupling import apply_particle
from src.pso import PSO
from experiments.configs import setting
from experiments.runner import run_one, _save_artifact, select_training
from experiments.validation import validate_complete_suite


class ValidationExtensionTests(unittest.TestCase):
    def test_missing_setting_and_altered_configuration(self):
        configs=[setting('first'),setting('second')]
        rows=[dict(name=c['name'],seed=s,config_json=json.dumps(c),train_accuracy=.5)
              for c in configs for s in range(10)]
        frame=pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs.csv'
            path.with_suffix('.metadata.json').write_text(json.dumps(dict(configs=configs,seeds=list(range(10)))))
            validate_complete_suite(frame,path)
            missing=frame[frame.name=='first']
            missing.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'missing='):
                select_training(path,Path(tmp)/'chosen.json')
            altered=frame.copy()
            altered.loc[0,'config_json']=json.dumps(setting('first',alpha=.1))
            with self.assertRaisesRegex(ValueError,'differs from metadata'):
                validate_complete_suite(altered,path)
            with self.assertRaisesRegex(ValueError,'seeds 0'):
                validate_complete_suite(frame.iloc[:-1],path)

    def test_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Missing suite metadata'):
                validate_complete_suite(pd.DataFrame(),Path(tmp)/'runs.csv')

    def test_reflection_positions_and_velocity_parity(self):
        solver=PSO(5,lambda x:0.,swarm_size=1,n_informants=1,
                   bounds=(-1,1),boundary='reflect')
        positions=np.array([[-1.25,1.25,3.25,-3.25,.25]])
        velocities=np.ones_like(positions)
        repaired=solver._bound(positions,velocities,np.random.default_rng(0))
        np.testing.assert_allclose(repaired,[[-.75,.75,-.75,.75,.25]])
        np.testing.assert_allclose(velocities,[[-1,-1,1,1,1]])

    def test_clip_and_reset_only_modify_escaping_components(self):
        for mode in ('clip','reset'):
            solver=PSO(3,lambda x:0.,swarm_size=1,n_informants=1,
                       bounds=(-1,1),boundary=mode)
            velocities=np.array([[2.,3.,4.]])
            result=solver._bound(np.array([[-2.,.25,2.]]),velocities,np.random.default_rng(0))
            self.assertEqual(result[0,1],.25)
            np.testing.assert_equal(velocities,[[0,3,0]])
            if mode=='clip': np.testing.assert_equal(result,[[-1,.25,1]])
            self.assertTrue(np.all((result>=-1)&(result<=1)))

    def test_inertia_schedule_controls_velocity(self):
        # With acceleration disabled, successive velocity ratios must equal
        # the requested inertia endpoints and midpoint: .9, .6, .3.
        solver=PSO(2,lambda x:0.,swarm_size=1,n_informants=1,
                   alpha=.9,alpha_end=.3,beta=0,gamma=0,delta=0,epsilon=0)
        observed=[]
        original=solver._bound
        def record(x,v,rng):
            observed.append(v.copy())
            return original(x,v,rng)
        with patch.object(solver,'_bound',side_effect=record): solver.run(4)
        rng=np.random.default_rng(0)
        rng.uniform(solver.low,solver.high,(1,2))
        initial=rng.uniform(-.1*(solver.high-solver.low),.1*(solver.high-solver.low),(1,2))
        np.testing.assert_allclose(observed[0],initial*.9)
        np.testing.assert_allclose(observed[1],initial*.9*.6)
        np.testing.assert_allclose(observed[2],initial*.9*.6*.3)

    def test_synthetic_runner_artifact_reconstructs_predictions(self):
        X=np.array([[0.,0,0,0],[1.,1,1,1],[-1.,-1,-1,-1],[.5,.5,.5,.5]])
        y=np.array([0,1,0,1])
        scaler=StandardScaler().fit(X)
        for encoded in (False,True):
            config=setting('synthetic',swarm_size=3,iterations=2,encode_activations=encoded)
            # Count distinct prediction calls made with the held-out array.
            held_out=X.copy()
            original=ANN.predict
            test_calls=[]
            def predict(ann,values):
                if values is held_out: test_calls.append(1)
                return original(ann,values)
            with patch.object(ANN,'predict',predict):
                row,artifact=run_one(config,0,(X,held_out,y,y,scaler))
            self.assertEqual(len(test_calls),1)
            self.assertEqual(row['evaluations'],6)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.npz'
                _save_artifact(path,artifact)
                with np.load(path,allow_pickle=False) as stored:
                    restored=ANN(config['layer_sizes'],config['activations'])
                    apply_particle(restored,stored['best_position'],encoded)
                    self.assertEqual(restored.activations,json.loads(row['chosen_activations_json']))
                    self.assertEqual(float(np.mean(restored.predict(held_out)==y)),row['test_accuracy'])
                    np.testing.assert_allclose(stored['scaler_mean'],scaler.mean_)
                    np.testing.assert_allclose(stored['scaler_scale'],scaler.scale_)
