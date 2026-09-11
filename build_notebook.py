"""Write an executable notebook using only the committed observed inputs/results."""
import json
from pathlib import Path

def build(path='factor_model_analysis.ipynb'):
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':[s]})
    def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[s]})
    md('# Real-observation equity and hiring research\nNo generated data. The hiring signal is occupation-wide and current-vintage, not company-specific historical postings. Read reports for limitations.')
    code("from real_research.analyze import load, features\nimport pandas as pd\np, f, h, manifest = load('data/real')\nprint(manifest['data_mode'], len(p), len(f), len(h))")
    md('## Filed information and factor features')
    code("panel, returns, hiring = features(p, f, h)\nassert (panel.fundamental_available_at < panel.date).all()\nprint(panel.tail())")
    md('## Computed portfolio results')
    code("performance = pd.read_csv('reports/performance.csv')\nprint(performance[performance.split == 'test'].to_string(index=False))")
    md('## Actual reported revenue forecasts')
    code("print(pd.read_csv('reports/forecast_scores.csv').to_string(index=False))\npred = pd.read_csv('reports/revenue_predictions.csv', parse_dates=['date','max_training_label'])\nassert (pred.max_training_label < pred.date).all()")
    md('## Limits\nSurviving-company selection, revised hiring history, small macro sample, and Q4 mixed-vintage derivations remain. The former generated-data results are withdrawn, not relabeled as real.')
    for i,c in enumerate(cells):c['id']=f'cell-{i}'
    Path(path).write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n')

if __name__=='__main__':build()
