<h1 align="left">Evaluation-as-Service</h1>

<p align="left"> EMNLP19 Submission: "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance" </p>

<h2 align="left">What is MoverScore and EvalSerivce</h2>

**MoverScore** measures semantic distance between system and reference texts by aligning semantically similar words and finding the corresponding travel costs.

**EvalSerivce** is a evaluation framework for NLG tasks, assigning scores (e.g., ROUGE ans MoverScore) to system-generated text by comparing it against human references for content matching.

<h2 align="left">Installation</h2>

Install the server and client via `pip`. They can be installed separately or even on *different* machines:
```bash
cd server/
python setup.py install # server
cd client/
python setup.py install # client
```

Note that the server MUST be running on **Python >= 3.5**. Again, the server does not support Python 2!

:point_up: The client can be running on both Python 2 and 3 [for the following consideration](#q-can-i-run-it-in-python-2).

<h2 align="center">Getting Started</h2>

#### 1. Download the data and Word-embeddings 
Download a model listed below, then uncompress the zip file into some folder

#### 2. Start the SummEval service
After installing the server, you should be able to use `summ-eval-start` CLI as follows:
```bash
summ-eval -eval_dir ../../ -num_worker=1 
```
This will start a service with one workers, meaning that it can handle up to four **concurrent** requests. More concurrent requests will be queued in a load balancer. Details can be found in our [FAQ](#q-what-is-the-parallel-processing-model-behind-the-scene).

#### 3. Use Client to Get Evaluation scores
Now you can encode sentences simply as follows:
```python
from summ_eval.client import EvalClient
sc = EvalClient()
sc.eval([[['This is test summary'], [['This is ref summary one'],['This is ref summary two']], 'rouge_n'], [['This is test summary two'], [['This is ref summary two'],['This is ref summary two']], 'rouge_n']])
{'0': [0.6, 0.8000000000000002]}
```

Below shows what the server looks like while encoding:
<p align="center"><img src=".github/server-run-demo.gif?raw=true"/></p>

#### Use Summ Eval Service Remotely
One may also start the service on one machine and call it from another machine as follows:

```python
# on another CPU machine
from summ_eval.client import EvalClient
sc = EvalClient(ip='xx.xx.xx.xx')  # ip address of the GPU machine
sc.eval([[['This is test summary'], [['This is ref summary one'],['This is ref summary two']], 'rouge_n'], [['This is test summary two'], [['This is ref summary two'],['This is ref summary two']], 'rouge_n']])
{'0': [0.6, 0.8000000000000002]}
```
### Client API


| Argument | Type | Default | Description |
|----------------------|------|-----------|-------------------------------------------------------------------------------|
| `ip` | str | `localhost` | IP address of the server |
| `port` | int | `5555` | port for pushing data from client to server, *must be consistent with the server side config* |
| `port_out` | int | `5556`| port for publishing results from server to client, *must be consistent with the server side config* |
| `output_fmt` | str | `ndarray` | the output format of the sentence encodes, either in numpy array or python List[List[float]] (`ndarray`/`list`) |
| `show_server_config` | bool | `False` | whether to show server configs when first connected |
| `check_version` | bool | `True` | whether to force client and server to have the same version |
| `identity` | str | `None` | a UUID that identifies the client, useful in multi-casting |
| `timeout` | int | `-1` | set the timeout (milliseconds) for receive operation on the client |

A `EvalClient` implements the following methods and properties:

| Method |  Description |
|--------|------|
|`.eval()`|Evaluate the summary and the reference text |
|`.close()`|Gracefully close the connection between the client and the server|
|`.status`|Get the client status in JSON format|
|`.server_status`|Get the server status in JSON format|

