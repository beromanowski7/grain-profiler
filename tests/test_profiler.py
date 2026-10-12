import pandas as pd
from profiler import theils_u, dependency, key_candidates, heldout_lookup, analyze


def sample():
    return pd.DataFrame({'Warehouse': ['A','B']*20, 'Region': ['North','South']*20,
                         'Status': ['Ready','Hold','Ready','Ready']*10,
                         'Qty': list(range(40))})


def test_directional():
    df = pd.DataFrame({'Warehouse':['A','B','C','A','B','C'],
                       'Region':['N','N','S','N','N','S']})
    assert abs(theils_u(df.Warehouse, df.Region)-1) < 1e-10
    assert theils_u(df.Region, df.Warehouse) < 1


def test_exact_and_keys():
    df = sample()
    assert dependency(df, 'Warehouse', 'Region')['exact']
    keys, _, _ = key_candidates(df, ['Warehouse','Region','Qty'])
    assert ('Qty',) in keys['columns'].tolist()


def test_heldout():
    r = heldout_lookup(sample(), 'Warehouse', 'Region')
    assert r['accuracy'] == 1
    assert r['key_coverage'] == 1


def test_analyze():
    r = analyze(sample())
    assert len(r['profile']) == 4
    assert 'Warehouse' in r['matrix'].columns
