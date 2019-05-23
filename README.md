<h1 align="left">Evaluation-as-Service</h1>

<p align="left"> EMNLP19 Submission: "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance" </p>

<h2 align="left">What is MoverScore and EvalSerivce</h2>

**MoverScore** measures semantic distance between system and reference texts by aligning semantically similar words and finding the corresponding travel costs.

**EvalSerivce** is a evaluation framework for NLG tasks, assigning scores (e.g., ROUGE ans MoverScore) to system-generated text by comparing it against human references for content matching.

<h2 align="left">Installation</h2>

Install the server and client via `pip`. They can be installed separately or even on *different* machines:
```bash
cd server/
python3 setup.py install # server
cd client/
python3 setup.py install # client
```

Note that the server MUST be running on **Python >= 3.5**. Again, the server does not support Python 2!

The client can be running on both Python 2 and 3 [for the following consideration](#q-can-i-run-it-in-python-2).

<h2 align="left">Getting Started</h2>

#### 1. Start the evaluation service
After installing the server, you should start a serivce as follows:
```bash
summ-eval-start -data_dir ../../ -num_worker=4
```
This will start the service with four workers, meaning that it can handle up to four **concurrent** requests.

#### 2. Use Client to Get Evaluation scores
Now you can get scores:
```python
from nlg_eval.client import EvalClient
ec = EvalClient()
ec.eval([[['This is test summary'], [['This is ref summary one'],['This is ref summary two']], 'rouge_n'], 
        [['This is test summary two'], [['This is ref summary two'],['This is ref summary two']], 'rouge_n']])
{'0': [0.6, 0.8000000000000002]}
```
