"""Motor TRACE MVP: classificação, decomposição linear exata e projeção STL exploratória."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def demo_data(n: int = 500, seed: int = 137) -> pd.DataFrame:
    """Dados inteiramente SINTÉTICOS, sem ligação ao German Credit Dataset."""
    rng = np.random.default_rng(seed)
    years = rng.integers(18, 68, n)
    amount = rng.integers(1500, 30000, n)
    duration = rng.choice([12, 24, 36, 48, 60], n)
    previous = rng.choice([0, 1, 2, 3], n, p=[.62, .22, .11, .05])
    savings = np.maximum(0, rng.normal(5500, 4200, n)).round()
    stability = rng.choice(['Estável', 'Variável', 'Temporário'], n, p=[.52,.31,.17])
    z = (-2.1 + amount/30000*1.3 + duration/60*.9 + previous*.95
         - savings/15000*1.15 + np.where(stability=='Temporário',.8,0)
         + np.where(stability=='Estável',-.6,0) + (years<23)*.2)
    prob = 1/(1+np.exp(-z))
    target = np.where(rng.random(n)<prob, 'Incumprimento', 'Cumprimento')
    return pd.DataFrame({'Idade':years,'Montante solicitado':amount,
        'Duração do empréstimo':duration,'Histórico de incumprimento':previous,
        'Poupança disponível':savings,'Estabilidade laboral':stability,
        'Resultado':target})


def train(data: pd.DataFrame, target: str):
    if target not in data.columns:
        raise ValueError('Coluna de resultado inexistente.')
    df = data.dropna(subset=[target]).copy()
    y = df[target].astype(str)
    if y.nunique()!=2:
        raise ValueError('O MVP requer exatamente duas classes na coluna de resultado.')
    X = df.drop(columns=[target]).copy()
    # Apenas tipos numéricos ou categorias; descartar identificadores óbvios.
    ids = [c for c in X.columns if str(c).lower().strip() in ('id','identificador','index','índice')]
    X = X.drop(columns=ids)
    X = X.dropna(axis=1, how='all')
    if X.shape[1]==0 or len(X)<50:
        raise ValueError('São necessárias pelo menos 50 linhas e uma variável preditora válida.')
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    X[cats] = X[cats].astype('string').fillna('Em falta').astype(str)
    if any(X[c].nunique(dropna=True)>60 for c in cats):
        raise ValueError('Existe uma variável categórica com mais de 60 valores distintos; remova identificadores ou texto livre.')
    transformers=[]
    if nums: transformers.append(('num',Pipeline([('fill', SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
    if cats: transformers.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats))
    pre=ColumnTransformer(transformers,remainder='drop',verbose_feature_names_out=True)
    model=Pipeline([('features',pre),('classifier',LogisticRegression(max_iter=1200,random_state=137))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=137,stratify=y)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    proba=model.predict_proba(Xte)[:,1]
    metrics={'acuracia_teste':float(accuracy_score(yte,pred)),
             'auc_teste':float(roc_auc_score((yte==model.classes_[1]).astype(int),proba)),
             'n_treino':len(Xtr),'n_teste':len(Xte),'classe_positiva':str(model.classes_[1])}
    return model,X,metrics,nums,cats


def explain(model, row: pd.DataFrame, nums: list[str], cats: list[str]):
    """Decomposição do log-odds de regressão logística: intercepto + soma de contribuições."""
    pre=model.named_steps['features']
    clf=model.named_steps['classifier']
    transformed=np.asarray(pre.transform(row))[0]
    contributions=transformed * clf.coef_[0]
    group={str(c):0.0 for c in row.columns}
    k=0
    if nums:
        for c in nums:
            group[str(c)]+=float(contributions[k]);k+=1
    if cats:
        encoder=pre.named_transformers_['cat']
        for c,values in zip(cats,encoder.categories_):
            width=len(values)
            group[str(c)]+=float(contributions[k:k+width].sum());k+=width
    intercept=float(clf.intercept_[0]); linear=intercept+sum(group.values())
    p=1/(1+math.exp(-max(-700,min(700,linear))))
    return group,intercept,linear,p


def stl_projection(contrib:dict[str,float]):
    """Projeção cromática experimental, não validada para decisões financeiras."""
    def category(name):
        x=name.lower()
        if any(t in x for t in ['histórico','historico','incumpr','default','arrears']):return 'Pressão financeira'
        if any(t in x for t in ['duração','duracao','montante','amount','duration','loan']):return 'Exposição temporal'
        if any(t in x for t in ['poupança','poupanca','savings','saldo']):return 'Proteção financeira'
        if any(t in x for t in ['estabilidade','emprego','laboral','job','employment']):return 'Estabilidade contextual'
        return 'Outras variáveis'
    palette={'Pressão financeira':'#F39B8E','Exposição temporal':'#E9B488',
             'Proteção financeira':'#80D4C6','Estabilidade contextual':'#7EA2DF',
             'Outras variáveis':'#BCC7D5'}
    totals={key:0.0 for key in palette}
    for name,value in contrib.items():totals[category(name)]+=abs(value)
    denominator=sum(totals.values()) or 1
    weights={key:round(value/denominator*100,1) for key,value in totals.items()}
    rgb=np.zeros(3)
    for name,w in weights.items():
        color=palette[name].lstrip('#')
        rgb+=w/100 * np.array([int(color[i:i+2],16) for i in (0,2,4)])
    hexcolor='#'+''.join(f'{int(round(v)):02X}' for v in np.clip(rgb,0,255))
    return weights,palette,hexcolor
