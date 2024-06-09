Relevant experimental results can be found in the `exp` directory.

To produce the best result. You can run Distilled-GPT2 by 

```bash
python distilgpt2.py
```
If you can't connect to HuggingFace, you can download the model locally, and `from_pretrained('path/to/your/model')`

Besides Distilled-GPT2, you can get the second best result by

```bash
bash run_lstm.sh
```
Note that the the parameter size may slightly exceed 60M, you can modify emsize and nhid to 800 if you want to strictly limit the parameters within 60M.
