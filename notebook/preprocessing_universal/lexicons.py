"""Local lexicon adapters used by the step-by-step notebook builder."""
from pathlib import Path
import hashlib
import pandas as pd
from stepwise import SLANG, PROTECTED, AMBIGUOUS

def read_lexicon(path, variant='variant', canonical='canonical', **read_csv_options):
    """Adapter for Kamus Alay/Colloquial/Combined/custom CSV; no forced approval."""
    lex = pd.read_csv(path, dtype=str, keep_default_na=False, **read_csv_options)
    lex = lex.rename(columns={variant: 'variant', canonical: 'canonical'})
    if not {'variant', 'canonical'} <= set(lex):
        raise ValueError('Butuh kolom variant dan canonical, atau atur nama kolom adapter.')
    for c in ['variant', 'canonical']:
        lex[c] = lex[c].str.strip().str.casefold()
    for c, default in {'source': Path(path).name, 'category': 'slang', 'language': 'id',
                       'confidence': '', 'review_status': 'pending'}.items():
        if c not in lex:
            lex[c] = default
    return lex


def approved_slang(lex):
    multiple = lex.groupby('variant').canonical.nunique()
    eligible = lex.variant.map(multiple).eq(1) & lex.review_status.eq('approved')
    eligible &= ~lex.variant.isin(PROTECTED | AMBIGUOUS) & lex.variant.str.fullmatch(r'[^\W\d_]+')
    eligible &= lex.canonical.ne('')
    return {**SLANG, **dict(lex.loc[eligible, ['variant', 'canonical']].itertuples(index=False, name=None))}


RESOURCE_URLS = {
    'colloquial.csv': 'https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv',
    'kamusalay.csv': 'https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv',
    'combined.json': 'https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/master/combined_slang_words.txt',
}


def external_lexicons(folder, download=False):
    """Prepare online once; reuse offline. All community mappings remain pending.
    Original resource files stay local; review redistribution terms at source.
    """
    import json
    import urllib.request
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, url in RESOURCE_URLS.items():
        path = folder / name
        if download and not path.exists():
            data = urllib.request.urlopen(url, timeout=45).read()
            tmp = path.with_suffix('.part')
            tmp.write_bytes(data)
            tmp.replace(path)
        if not path.exists():
            continue
        if name == 'combined.json':
            values = json.loads(path.read_text(encoding='utf-8'))
            table = pd.DataFrame(values.items(), columns=['variant', 'canonical'])
        elif name == 'kamusalay.csv':
            table = pd.read_csv(path, names=['variant', 'canonical'], encoding='latin-1', keep_default_na=False)
        else:
            table = pd.read_csv(path, keep_default_na=False).rename(columns={'slang': 'variant', 'formal': 'canonical'})
        table = table[['variant', 'canonical']].copy()
        table['source'] = url
        table['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        table['category'], table['language'] = 'slang', 'id'
        table['confidence'], table['review_status'] = '', 'pending'
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=['variant', 'canonical', 'source', 'source_sha256', 'category', 'language', 'confidence', 'review_status'])
    lex = pd.concat(tables, ignore_index=True).drop_duplicates(['variant', 'canonical', 'source'])
    for c in ['variant', 'canonical']:
        lex[c] = lex[c].str.strip().str.casefold()
    lex['conflict'] = lex.variant.map(lex.groupby('variant').canonical.nunique()).gt(1)
    return lex
