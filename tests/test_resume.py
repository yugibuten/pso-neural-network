"""Tests for interrupted and resumed experiment suites, using mocked runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.runner import run_suite, _atomic_write
from experiments.configs import setting


class ResumeTests(unittest.TestCase):
    def test_interruption_and_resume(self):
        config=setting('mock')
        calls=[]
        def fake_run(config,seed,data):
            calls.append(seed)
            if seed==1 and len(calls)==2:
                raise OSError(28,'No space left on device')
            return (dict(name=config['name'],seed=seed,config_json=json.dumps(config)),
                    dict(best_position=np.zeros(49),scaler_mean=np.zeros(4),
                         scaler_scale=np.ones(4),config_json=json.dumps(config),
                         seed=seed,history=np.zeros(1)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dataset=root/'data.txt'; dataset.write_text('synthetic')
            out=root/'runs.csv'
            with patch('experiments.runner.prepare_data',return_value=None), patch('experiments.runner.run_one',side_effect=fake_run):
                with self.assertRaises(OSError):
                    run_suite([config],out,dataset,seeds=[0,1,2])
                self.assertEqual(pd.read_csv(out).seed.tolist(),[0])
                run_suite([config],out,dataset,seeds=[0,1,2],resume=True)
                self.assertEqual(calls,[0,1,1,2])
                self.assertEqual(pd.read_csv(out).seed.tolist(),[0,1,2])
                run_suite([config],out,dataset,seeds=[0,1,2],resume=True)
                self.assertEqual(calls,[0,1,1,2])
                with self.assertRaises(ValueError):
                    run_suite([setting('mock',alpha=.5)],out,dataset,seeds=[0,1,2],resume=True)

    def test_failed_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'runs.csv'; out.write_text('preserved')
            def fail(path):
                path.write_text('partial')
                raise OSError(28,'No space left on device')
            with self.assertRaises(OSError): _atomic_write(out,fail)
            self.assertEqual(out.read_text(),'preserved')
            self.assertEqual(list(Path(tmp).iterdir()),[out])
