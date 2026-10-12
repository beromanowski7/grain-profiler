"""GRAIN: exploratory, sample-based structural diagnostics (not causal proof)."""
from itertools import combinations
import numpy as np
import pandas as pd

MISSING = '<GRAIN:MISSING>'


def categories(s):
    """Convert to stable labels, including a distinct missing-value category."""
    return s.map(lambda v: MISSING if pd.isna(v) else f'{type(v).__name__}:{v}')


def entropy(s):
    p = categories(s).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) else np.nan


def theils_u(x, y):
    """Fraction of target Y entropy explained by X; NaN for constant Y."""
    if len(x) == 0:
        return np.nan
    hx = categories(x)
    hy = categories(y)
    p = hy.value_counts(normalize=True)
    h_y = float(-(p * np.log2(p)).sum())
    if h_y == 0:
        return np.nan
    tab = pd.crosstab(hx, hy)
    joint = tab.to_numpy(dtype=float)
    row_totals = joint.sum(axis=1)
    conditional = 0.0
    for row, count in zip(joint, row_totals):
        probs = row[row > 0] / count
        conditional += (count / len(x)) * float(-(probs * np.log2(probs)).sum())
    return float(np.clip(1 - conditional / h_y, 0, 1))


def dependency(df, x, y):
    pairs = pd.DataFrame({'x': categories(df[x]), 'y': categories(df[y])})
    if pairs.empty:
        return {'exact': False, 'violating_keys': 0, 'keys': 0, 'row_agreement': np.nan}
    tab = pairs.groupby(['x', 'y'], dropna=False).size().unstack(fill_value=0)
    violations = int((tab.gt(0).sum(axis=1) > 1).sum())
    return {'exact': violations == 0, 'violating_keys': violations,
            'keys': len(tab), 'row_agreement': float(tab.max(axis=1).sum() / len(pairs))}


def column_profile(df):
    rows = []
    for name in df.columns:
        s = df[name]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        dates = pd.api.types.is_datetime64_any_dtype(s)
        nonnull = s.dropna()
        counts = categories(s).value_counts()
        row = {'column': name, 'dtype': str(s.dtype), 'rows': len(s),
               'missing_pct': round(100 * s.isna().mean(), 2),
               'distinct': int(s.nunique(dropna=True)),
               'uniqueness_pct': round(100 * s.nunique(dropna=True) / max(len(nonnull), 1), 2),
               'top_frequency_pct': round(100 * counts.iloc[0] / len(s), 2) if len(s) else np.nan,
               'entropy_bits': entropy(s),
               'min': None, 'max': None, 'mean': np.nan,
               'zero_pct': np.nan, 'negative_pct': np.nan,
               'first_date': None, 'last_date': None}
        if numeric and len(nonnull):
            row.update(min=float(nonnull.min()), max=float(nonnull.max()),
                       mean=float(nonnull.mean()), zero_pct=round(100 * s.eq(0).mean(), 2),
                       negative_pct=round(100 * s.lt(0).mean(), 2))
        if dates and len(nonnull):
            row.update(first_date=str(nonnull.min()), last_date=str(nonnull.max()))
        rows.append(row)
    return pd.DataFrame(rows)


def relationship_report(df, columns):
    mat = pd.DataFrame(np.nan, index=columns, columns=columns)
    audits = []
    for x in columns:
        for y in columns:
            if x == y:
                continue
            u = theils_u(df[x], df[y])
            mat.loc[x, y] = u
            d = dependency(df, x, y)
            audits.append({'predictor': x, 'target': y, 'theils_u': u, **d,
                           'predictor_uniqueness_pct': 100 * df[x].nunique(dropna=False) / max(len(df), 1)})
    return mat, pd.DataFrame(audits).sort_values(['exact', 'theils_u'], ascending=False)


def key_candidates(df, columns, max_size=3, max_combinations=3000):
    """Find minimal unique column combinations; sample uniqueness is not business-key proof."""
    keys = []
    checked = 0
    for size in range(1, min(max_size, len(columns)) + 1):
        for combo in combinations(columns, size):
            if checked >= max_combinations:
                return pd.DataFrame(keys), checked, True
            checked += 1
            if any(set(k['columns']).issubset(combo) for k in keys):
                continue
            unique_rows = df[list(combo)].drop_duplicates().shape[0]
            if unique_rows == len(df) and len(df):
                keys.append({'columns': combo, 'column_count': size,
                             'unique_rows': unique_rows, 'total_rows': len(df)})
    return pd.DataFrame(keys), checked, False


def heldout_lookup(df, x, y, test_fraction=.25, seed=42, chronological=None):
    """Simple majority-per-key classifier. For dates, train earlier rows, test later rows."""
    if len(df) < 8 or df[y].nunique(dropna=False) < 2:
        return None
    ordered = df.copy()
    if chronological:
        ordered = ordered.sort_values(chronological, kind='stable')
    else:
        ordered = ordered.sample(frac=1, random_state=seed)
    ntrain = max(1, min(len(ordered)-1, int(len(ordered)*(1-test_fraction))))
    train, test = ordered.iloc[:ntrain], ordered.iloc[ntrain:]
    tx, ty = categories(train[x]), categories(train[y])
    vx, vy = categories(test[x]), categories(test[y])
    mapping = pd.DataFrame({'x': tx, 'y': ty}).groupby('x')['y'].agg(lambda z: z.value_counts().index[0])
    majority = ty.value_counts().index[0]
    raw = vx.map(mapping)
    predictions = raw.fillna(majority)
    return {'predictor': x, 'target': y, 'test_rows': len(test),
            'accuracy': float((predictions == vy).mean()),
            'baseline_accuracy': float((vy == majority).mean()),
            'lift_percentage_points': float(100*((predictions == vy).mean() - (vy == majority).mean())),
            'key_coverage': float(raw.notna().mean()),
            'split': f'chronological ({chronological})' if chronological else 'random (seed=42)'}


def analyze(df, max_cols=25, max_key_size=3, chronological=None):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    if not df.columns.is_unique:
        raise ValueError('Duplicate column names: rename columns before profiling.')
    if df.empty:
        raise ValueError('The selected sheet has no data rows.')
    profile = column_profile(df)
    cols = [c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c])
            and (not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 20)]
    cols = cols[:max_cols]
    matrix, deps = relationship_report(df, cols)
    keys, checked, truncated = key_candidates(df, list(df.columns)[:max_cols], max_key_size)
    validations = []
    for _, row in deps.head(100).iterrows():
        result = heldout_lookup(df, row['predictor'], row['target'], chronological=chronological)
        if result:
            validations.append(result)
    return {'profile': profile, 'matrix': matrix, 'dependencies': deps,
            'keys': keys, 'validation': pd.DataFrame(validations),
            'key_combinations_checked': checked, 'key_search_truncated': truncated,
            'columns_in_matrix': cols}
