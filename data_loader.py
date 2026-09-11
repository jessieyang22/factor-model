"""Real-source entry points. Generated hiring factors are no longer supported."""
from real_research.download import download
from real_research.analyze import load

def load_real_inputs(root='data/real'):
    return load(root)

def load_hiring_momentum_factor(*args,**kwargs):
    raise ValueError('Company-specific hiring return factors require observed postings AND subsequent stock returns. Use the real aggregate-hiring study; no proxy return is generated.')

if __name__=='__main__':download()
