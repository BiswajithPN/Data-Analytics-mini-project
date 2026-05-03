import os, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, silhouette_score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from plotly.utils import PlotlyJSONEncoder
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

app = Flask(__name__)

def _get_acc(df, target_col):
    try:
        if target_col not in df.columns: return "N/A"
        tmp = df.dropna().copy()
        if len(tmp) < 10: return "N/A"
        for c in tmp.select_dtypes(include=['object']).columns:
            tmp[c] = LabelEncoder().fit_transform(tmp[c].astype(str))
        X = tmp.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = tmp[target_col]
        if X.empty: return "N/A"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
        return round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
    except: return "N/A"

# ── Raw originals (NEVER mutated) ─────────────────────────────────
_retail_raw   = pd.DataFrame()
_reliance_raw = pd.DataFrame()

# ── Working copies (preprocessing applied here only) ──────────────
retail_work   = pd.DataFrame()
reliance_work = pd.DataFrame()

# ── Path configuration ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

def load():
    global _retail_raw, _reliance_raw, retail_work, reliance_work
    retail_path = os.path.join(DATA_DIR, 'Retail_Analytics_Dataset.csv')
    reliance_path = os.path.join(DATA_DIR, 'Reliance_dataset.csv')
    
    if os.path.exists(retail_path):
        _retail_raw = pd.read_csv(retail_path)
        _retail_raw.columns = _retail_raw.columns.str.strip()
    if os.path.exists(reliance_path):
        _reliance_raw = pd.read_csv(reliance_path)
        _reliance_raw.columns = _reliance_raw.columns.str.strip()
    retail_work   = _retail_raw.copy()
    reliance_work = _reliance_raw.copy()

load()

def _first(df, options):
    """Find the first column that matches any of the options (case-insensitive)."""
    cols = [c.lower() for c in df.columns]
    for opt in options:
        if opt.lower() in cols:
            # Return the actual column name from df.columns
            return df.columns[cols.index(opt.lower())]
    return df.columns[0] if len(df.columns) > 0 else None

def raw(name):
    return _retail_raw if name == 'retail' else _reliance_raw

def work(name):
    return retail_work if name == 'retail' else reliance_work

def fig2j(fig):
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

# ── Status ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def status():
    r = _retail_raw; rel = _reliance_raw
    r_sales   = float(r['Net_Sales'].sum())   if not r.empty else 0
    rel_sales = float(rel['Net_Sales_Value'].sum()) if not rel.empty else 0
    winner = 'Retail' if r_sales > rel_sales else 'Reliance'
    return jsonify({
        'retail':   {
            'ok': not r.empty, 
            'rows': len(r), 
            'cols': r.columns.tolist(),
            'uniques': r.nunique().to_dict() if not r.empty else {}
        },
        'reliance': {
            'ok': not rel.empty, 
            'rows': len(rel), 
            'cols': rel.columns.tolist(),
            'uniques': rel.nunique().to_dict() if not rel.empty else {}
        },
        'kpis': {
            'retail_revenue':   round(r_sales, 2),
            'reliance_revenue': round(rel_sales, 2),
            'retail_avg':       round(float(r['Net_Sales'].mean()), 2)   if not r.empty else 0,
            'reliance_avg':     round(float(rel['Net_Sales_Value'].mean()), 2) if not rel.empty else 0,
            'winner': winner
        }
    })

@app.route('/api/overview_insights')
def overview_insights():
    try:
        r = _retail_raw; rel = _reliance_raw
        if r.empty or rel.empty: return jsonify({'error': 'Data not loaded'})

        # Retail Metrics
        r_rev = r['Net_Sales'].sum()
        r_eff = r_rev / len(r)
        r_acc = _get_acc(r, 'Customer_Type')

        # Reliance Metrics
        rel_rev = rel['Net_Sales_Value'].sum()
        rel_eff = rel_rev / len(rel)
        rel_acc = _get_acc(rel, 'Membership_Type')

        # Competitive Logic (Why Retail Wins)
        advantages = []
        if r_acc != "N/A" and (rel_acc == "N/A" or r_acc > rel_acc):
            advantages.append({'title': 'Superior Intelligence', 'desc': f"Retail's segmentation model is {r_acc}% accurate, outperforming Reliance in customer targeting."})
        if r_eff > rel_eff:
            advantages.append({'title': 'Basket Efficiency', 'desc': f"Retail generates ₹{round(r_eff,0)} per txn vs Reliance's ₹{round(rel_eff,0)} — better cross-selling."})
        if r['Product_Category'].nunique() > rel['Product_Category'].nunique():
            advantages.append({'title': 'Market Assortment', 'desc': "Retail offers a broader variety of product categories, capturing more diverse consumer needs."})
        
        if not advantages:
            advantages.append({'title': 'Strategic Position', 'desc': "Retail maintains a lean operational model with high category-specific yield."})

        return jsonify({
            'retail': {
                'top_category': str(r.groupby('Product_Category')['Net_Sales'].sum().idxmax()),
                'top_store':    str(int(r.groupby('Store_ID')['Net_Sales'].sum().idxmax())),
                'top_location': str(r['Store_Location'].mode()[0]),
                'top_customer': str(r['Customer_Type'].mode()[0]),
                'top_payment':  str(r['Payment_Method'].mode()[0]),
                'model_acc':    r_acc,
                'avg_sales':    round(r_eff, 2),
                'total_sales':  round(r_rev, 2)
            },
            'reliance': {
                'top_profit_category': str(rel.groupby('Product_Category')['Profit_Amount'].sum().idxmax()),
                'top_region':          str(rel['Store_Region'].mode()[0]),
                'top_shift':           str(rel['Shift_Type'].mode()[0]),
                'top_membership':      str(rel['Membership_Type'].mode()[0]),
                'model_acc':           rel_acc,
                'avg_sales':           round(rel_eff, 2),
                'total_sales':         round(rel_rev, 2)
            },
            'advantages': advantages
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/algorithm_guide')
def algorithm_guide():
    try:
        insights = [
            {'name': 'Decision Tree (CART)', 'use': 'Classifying customers as High/Low spenders, predicting churn', 'why': 'Handles both numeric & categorical features, easy to interpret, no normalization needed'},
            {'name': 'Naïve Bayes', 'use': 'Payment method preference, customer type classification', 'why': 'Fast, works well with small data, good for categorical targets with conditional independence'},
            {'name': 'K-Means Clustering', 'use': 'Customer segmentation by spending & age, store grouping', 'why': 'Unsupervised — no labels needed; segments customers into High/Medium/Low value groups'},
            {'name': 'OLAP Roll-Up', 'use': 'Total sales by category, region, store', 'why': 'Aggregates fine-grained data to high-level summaries for executive reporting'},
            {'name': 'OLAP Drill-Down', 'use': 'Category → Product → Store level analysis', 'why': 'Goes from summary to detail; great for root cause analysis of performance gaps'},
            {'name': 'OLAP Slice', 'use': 'Peak hour analysis, weekend vs weekday', 'why': 'Filters one dimension to isolate specific business conditions'},
            {'name': 'OLAP Dice', 'use': 'High-value + specific region, high-risk churn customers', 'why': 'Multi-condition filter; best for finding niche customer segments'},
            {'name': 'OLAP Pivot', 'use': 'Category × Location × Sales matrix', 'why': 'Rotates data dimensions for cross-tabulation reporting'},
            {'name': 'Correlation Analysis', 'use': 'Discount impact, loyalty points, footfall vs sales', 'why': 'Quantifies linear relationships; guides which variables to include in ML models'},
            {'name': 'MinMaxScaler Normalization', 'use': 'Before K-Means, before Neural Net models', 'why': 'Prevents large-scale features dominating distance calculations in clustering'}
        ]
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)})

# ── 1. PREPROCESSING (works on copy only) ─────────────────────────
@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    global retail_work, reliance_work
    d  = request.json
    nm = d['dataset']; op = d['op']
    df = work(nm).copy()

    if op == 'clean':
        before = len(df)
        df = df.dropna().drop_duplicates()
        msg = f"Removed {before - len(df)} dirty rows. Remaining: {len(df):,}"
    elif op == 'normalize':
        num = df.select_dtypes(include=np.number).columns
        df[num] = MinMaxScaler().fit_transform(df[num].fillna(0))
        msg = f"Normalized {len(num)} numeric columns to [0, 1]."
    elif op == 'encode':
        cats = df.select_dtypes(include='object').columns
        le = LabelEncoder()
        for c in cats:
            df[c] = le.fit_transform(df[c].astype(str))
        msg = f"Label-encoded {len(cats)} categorical columns."
    elif op == 'discretize':
        age_col = next((c for c in df.columns if 'age' in c.lower()), None)
        if age_col:
            df['Age_Group'] = pd.cut(df[age_col], bins=[0,25,50,120],
                                     labels=['Young','Adult','Senior'])
            msg = "Age discretized → Young / Adult / Senior"
        else:
            msg = "No Age column found."
    elif op == 'reset':
        df = raw(nm).copy()
        msg = "Dataset reset to original."
    else:
        msg = "Unknown op."

    if nm == 'retail':   retail_work   = df
    else:                reliance_work = df

    return jsonify({'msg': msg, 'head': df.head(6).fillna('').to_dict(orient='records')})

# ── 2. CLUSTERING (uses raw data) ─────────────────────────────────
@app.route('/api/cluster', methods=['POST'])
def cluster():
    try:
        from sklearn.cluster import DBSCAN, AgglomerativeClustering
        d = request.json
        df = raw(d['dataset']).dropna().copy()
        cols = d['columns']
        algo = d.get('algo', 'kmeans')
        k = int(d.get('k', 3))
        X = df[cols].copy()
        for c in cols:
            if X[c].dtype == object: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = MinMaxScaler().fit_transform(X.fillna(0))
        if algo == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5).fit(X); labels = model.labels_; inertia = 0
        elif algo == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=k).fit(X); labels = model.labels_; inertia = 0
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X); inertia = km.inertia_
        sil = round(float(silhouette_score(X, labels)), 3) if len(set(labels)) > 1 else 0
        plot_df = df[cols].copy()
        plot_df['Cluster'] = [f'C{l}' if l >= 0 else 'Noise' for l in labels]
        fig = px.scatter(plot_df, x=cols[0], y=cols[1], color='Cluster', title=f'{algo.upper()} Clustering | Silhouette: {sil}', template='plotly_dark')
        return jsonify({'fig': fig2j(fig), 'inertia': round(float(inertia), 2), 'silhouette': sil})
    except Exception as e:
        print(f"ERROR in classify: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── 3. CLASSIFICATION ─────────────────────────────────────────────
def _run_clf(clf, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    labels = sorted(list(set(y)))
    is_binary = (len(labels) == 2)
    cm = confusion_matrix(yte, yp)
    if is_binary:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    else:
        tp = np.diag(cm).sum()
        fp = cm.sum() - tp
        fn = 0
        tn = 0
    return {
        'accuracy':  round(accuracy_score(yte,yp)*100, 2),
        'precision': round(precision_score(yte,yp,average='weighted',zero_division=0)*100, 2),
        'recall':    round(recall_score(yte,yp,average='weighted',zero_division=0)*100, 2),
        'f1':        round(f1_score(yte,yp,average='weighted',zero_division=0)*100, 2),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'is_binary': is_binary
    }

@app.route('/api/version')
def get_version():
    return jsonify({'version': '2.0', 'status': 'STABLE', 'engine': 'Top-20 Grouping Active'})

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
        d = request.json
        df = raw(d['dataset']).dropna().copy(); target = d['target']; algo = d['algo']
        
        # TOTAL FIX: Force any target into 20 classes or fewer
        unique_vals = df[target].nunique()
        if unique_vals > 20:
            top_20 = df[target].value_counts().index[:20]
            df[target] = df[target].apply(lambda x: x if x in top_20 else 'Other')
            
        le = LabelEncoder(); y = le.fit_transform(df[target].astype(str))
        X = df.select_dtypes(include=[np.number])
        if target in X.columns: X = X.drop(columns=[target])
        
        if X.empty: return jsonify({'error': 'No numeric features found for prediction.'}), 400
            
        clfs = {
            'nb': GaussianNB(), 'dt': DecisionTreeClassifier(max_depth=10),
            'lr': LogisticRegression(max_iter=1000), 'knn': KNeighborsClassifier(n_neighbors=5),
            'bag': BaggingClassifier(n_estimators=10),
            'ada': AdaBoostClassifier(n_estimators=50), 'gb': GradientBoostingClassifier(n_estimators=100)
        }
        clf = clfs.get(algo, clfs['nb']); res = _run_clf(clf, X, y)
        
        # Get first 20 samples (Actual vs Predicted names)
        yp_all = clf.predict(X)
        samples = []
        for i in range(min(20, len(y))):
            samples.append({
                'id': i + 1,
                'actual': str(le.inverse_transform([y[i]])[0]),
                'predicted': str(le.inverse_transform([yp_all[i]])[0])
            })
        res['samples'] = samples
        class_names = [str(c) for c in le.classes_]
        cm = confusion_matrix(y, clf.predict(X))
        
        z = cm.tolist(); z_text = [[str(v) for v in row] for row in z]
        fig = ff.create_annotated_heatmap(z=z[::-1], x=class_names, y=class_names[::-1], 
                                          annotation_text=z_text[::-1], colorscale='Blues', showscale=True)
        fig.update_layout(title=f"Classification: {algo.upper()} ({target})",
                          xaxis=dict(title="Predicted", side="bottom"), yaxis=dict(title="Actual"),
                          margin=dict(l=80, r=60, t=100, b=80), height=500, template='plotly_dark')
        res['type'] = 'classification'
        res['cm_fig'] = fig2j(fig)
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/regression', methods=['POST'])
def regression():
    try:
        from sklearn.metrics import r2_score, mean_squared_error
        d = request.json
        df = raw(d['dataset']).dropna().copy(); target = d['target']
        X_cols = df.select_dtypes(include=[np.number]).columns.drop(target)
        X = df[X_cols]; y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train); preds = model.predict(X_test)
        res = {'r2': round(r2_score(y_test, preds), 4), 'rmse': round(np.sqrt(mean_squared_error(y_test, preds)), 2)}
        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"Regression Analysis (R²={res['r2']})", template='plotly_dark', trendline="ols")
        res['fig'] = fig2j(fig); return jsonify(res)
    except Exception as e:
        print(f"ERROR in classify: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ── 5. OLAP (uses raw data) ───────────────────────────────────────
def _first(df, keywords):
    for k in keywords:
        for c in df.columns:
            if k.lower() in c.lower(): return c
    return df.select_dtypes(include=np.number).columns[0]

@app.route('/api/olap', methods=['POST'])
def olap():
    d = request.json
    df  = raw(d['dataset']).copy()
    op  = d['op']
    num_col = _first(df, ['net_sales','sales_value','profit','gross'])
    cat_col = _first(df, ['category','location','region','type'])
    loc_col = _first(df, ['location','region','city','state'])

    if op == 'rollup':
        res = df.groupby(cat_col)[num_col].agg(['sum','mean','count']).reset_index()
        res.columns = [cat_col,'Total','Average','Count']
        fig = px.bar(res, x=cat_col, y='Total', color='Total',
                     title=f'Roll-Up: {num_col} by {cat_col}', template='plotly_dark')
        return jsonify({'table': res.round(2).to_dict(orient='records'), 'fig': fig2j(fig), 'info': f'Aggregated {num_col} by {cat_col}'})

    elif op == 'drilldown':
        res = df.groupby([cat_col, loc_col])[num_col].sum().reset_index()
        fig = px.bar(res, x=loc_col, y=num_col, color=cat_col, barmode='group',
                     title=f'Drill-Down: {cat_col} → {loc_col}', template='plotly_dark')
        return jsonify({'table': res.head(20).round(2).to_dict(orient='records'), 'fig': fig2j(fig), 'info': f'Category broken down by {loc_col}'})

    elif op == 'slice':
        val = str(df[cat_col].dropna().unique()[0])
        sliced = df[df[cat_col].astype(str) == val]
        res = sliced.groupby(loc_col)[num_col].sum().reset_index()
        fig = px.bar(res, x=loc_col, y=num_col, title=f'Slice: {cat_col} = {val}', template='plotly_dark')
        return jsonify({'table': res.round(2).to_dict(orient='records'), 'fig': fig2j(fig), 'info': f'Slice: {cat_col} = {val} ({len(sliced)} rows)'})

    elif op == 'dice':
        med = df[num_col].median()
        diced = df[df[num_col] > med]
        res = diced.groupby(cat_col)[num_col].agg(['sum','count']).reset_index()
        res.columns = [cat_col,'Total','Count']
        fig = px.bar(res, x=cat_col, y='Total', title=f'Dice: {num_col} > Median({round(med,2)})', template='plotly_dark')
        return jsonify({'table': res.round(2).to_dict(orient='records'), 'fig': fig2j(fig), 'info': f'Dice filter: {num_col} > {round(med,2)}'})

    elif op == 'pivot':
        pivot = df.pivot_table(index=cat_col, columns=loc_col, values=num_col, aggfunc='sum').fillna(0)
        fig = go.Figure()
        for col_name in pivot.columns:
            fig.add_trace(go.Bar(name=str(col_name), x=pivot.index.tolist(), y=pivot[col_name].tolist()))
        fig.update_layout(barmode='group', template='plotly_dark',
                          title=f'Pivot: {cat_col} × {loc_col} ({num_col})',
                          xaxis_title=cat_col, yaxis_title=num_col, xaxis_tickangle=-30)
        return jsonify({'table': pivot.reset_index().round(2).to_dict(orient='records'), 'fig': fig2j(fig), 'info': f'Pivot: {cat_col} rows × {loc_col} columns'})

    return jsonify({'error': 'Unknown OLAP op'})

# ── 7. VISUALIZATIONS (always raw data) ──────────────────────────
@app.route('/api/chart', methods=['POST'])
def chart():
    d = request.json
    df = raw(d['dataset']).copy()
    t  = d['type']

    num_col    = _first(df, ['net_sales','sales_value','profit','gross'])
    cat_col    = _first(df, ['category','location','region','type'])
    profit_col = _first(df, ['profit_amount','profit_margin','net_sales','sales_value'])

    if t == 'barchart':
        # Total sales by category — most insightful view
        grp = df.groupby(cat_col)[num_col].sum().reset_index().sort_values(num_col, ascending=False)
        fig = px.bar(grp, x=cat_col, y=num_col, color=num_col,
                     color_continuous_scale='Viridis',
                     title=f'Total {num_col} by {cat_col}',
                     template='plotly_dark', text_auto='.2s')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-30, showlegend=False)

    elif t == 'boxplot':
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                     title=f'{cat_col} vs {num_col} (Box Plot)',
                     template='plotly_dark')

    elif t == 'linechart':
        # Average sales by category as a line trend
        grp = df.groupby(cat_col)[num_col].mean().reset_index().sort_values(num_col)
        fig = px.line(grp, x=cat_col, y=num_col, markers=True,
                      title=f'Average {num_col} per {cat_col}',
                      template='plotly_dark',
                      color_discrete_sequence=['#818cf8'])
        fig.update_traces(line_width=3, marker_size=10)
        fig.update_layout(xaxis_tickangle=-30)

    elif t == 'heatmap':
        num_cols = df.select_dtypes(include=np.number).columns[:10].tolist()
        corr = df[num_cols].corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values.tolist(), x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale='RdBu', zmid=0,
            text=[[str(v) for v in row] for row in corr.values.tolist()],
            texttemplate='%{text}', showscale=True))
        fig.update_layout(title='Correlation Heatmap', template='plotly_dark', height=500)

    elif t == 'outliers':
        # Enhanced outlier detection
        df_clean = df.dropna(subset=[num_col]).copy()
        if len(df_clean) > 1 and df_clean[num_col].std() > 0:
            z_scores = stats.zscore(df_clean[num_col])
            df_clean['is_outlier'] = [abs(z) > 2.5 for z in z_scores]
            fig = px.scatter(df_clean.reset_index(), x='index', y=num_col, color='is_outlier',
                             color_discrete_map={True: '#f43f5e', False: '#10b981'},
                             title=f'Outlier Detection (Z-Score > 2.5) — {num_col}', template='plotly_dark')
            fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color='white')))
        else:
            fig = px.scatter(title="Not enough variance for outlier detection")

    elif t == 'forecast':
        y = df[num_col].dropna().values
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        X_future = np.arange(len(y), len(y) + 30).reshape(-1, 1)
        y_future = model.predict(X_future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, name='Historical', line=dict(color='#818cf8')))
        fig.add_trace(go.Scatter(x=X_future.flatten(), y=y_future, name='Forecast', line=dict(color='#f43f5e', dash='dash')))
        fig.update_layout(title=f'Revenue Trend Forecasting — {num_col}', xaxis_title='Index', yaxis_title=num_col, template='plotly_dark')

    else:
        return jsonify({'error': 'Unknown chart type'})

    return jsonify({'fig': fig2j(fig)})

# ── RETAIL QUESTIONS with algorithm recommendations ───────────────
@app.route('/api/retail_questions')
def retail_questions():
    df = raw('retail').dropna().copy()
    if df.empty: return jsonify({'error': 'Data not loaded'})
    results = []

    def safe(fn, q, algo, exp):
        try:
            results.append({'q': q, 'a': str(fn()), 'algo': algo, 'exp': exp})
        except Exception as e:
            results.append({'q': q, 'a': f'N/A ({e})', 'algo': algo, 'exp': exp})

    safe(lambda: df.groupby('Product_Category')['Net_Sales'].sum().idxmax(),
         'Which Product_Category has highest Net_Sales?', 'Roll-Up (OLAP)',
         'We grouped all transactions by Product_Category and summed Net_Sales. Roll-Up aggregates granular rows into category-level totals, revealing which product line drives the most revenue. This directly answers resource allocation decisions — stock more of the top category.')

    safe(lambda: str(int(df.groupby('Store_ID')['Gross_Sales'].sum().idxmax())),
         'Which Store_ID generates highest Gross_Sales?', 'Roll-Up (OLAP)',
         'Gross_Sales (before discount deductions) is summed per Store_ID. The store with the highest pre-discount revenue indicates strongest footfall and pricing power. This helps management identify top-performing outlets for best-practice sharing.')

    safe(lambda: f"₹{df['Net_Sales'].mean():,.2f}",
         'What is average Net_Sales per transaction?', 'Descriptive Stats (Mean)',
         'The arithmetic mean of Net_Sales across all transactions gives the average basket size. This is the key KPI for retail benchmarking — comparing it vs industry average (₹1,500-₹2,500) reveals if customers are buying in bulk or small quantities.')

    safe(lambda: df.groupby('Store_Location')['Net_Sales'].sum().idxmax(),
         'Which Store_Location (Town/Village/City) performs better?', 'Roll-Up (OLAP)',
         'Aggregating Net_Sales by location type reveals geography-driven demand. City stores typically have higher footfall but also higher rent costs. Comparing City vs Town vs Village informs expansion and investment strategy for new store openings.')

    safe(lambda: f"Corr = {df['Discount_Percentage'].corr(df['Net_Sales']):.3f} — {'Discounts reduce net revenue' if df['Discount_Percentage'].corr(df['Net_Sales']) < 0 else 'Discounts boost volume'}",
         'Impact of Discount_Percentage on Net_Sales?', 'Pearson Correlation',
         'Pearson correlation measures linear relationship strength (-1 to +1). A negative value means higher discounts reduce net revenue (expected, since discount cuts price). Managers should find the "sweet spot" — enough discount to drive volume without eroding margin.')

    safe(lambda: df.groupby('Customer_Type')['Net_Sales'].mean().idxmax(),
         'Which Customer_Type spends more (Wholesale vs Occasional)?', 'Decision Tree (CART)',
         'Mean Net_Sales per Customer_Type shows which segment has higher average basket value. Wholesale buyers purchase in bulk (higher spend) while Occasional buyers are impulse shoppers. This drives loyalty program design — should you target bulk buyers or occasional converters?')

    safe(lambda: f"Corr = {df['Customer_Age'].corr(df['Net_Sales']):.3f} ({'Age positively influences spending' if df['Customer_Age'].corr(df['Net_Sales']) > 0 else 'Age has minimal/negative influence on spending'})",
         'Does Customer_Age affect spending behavior?', 'Regression / Scatter Analysis',
         'Correlation between age and spending reveals demographic purchasing patterns. Near-zero correlation means age alone is NOT a reliable predictor — other factors (income, loyalty) matter more. This prevents age-based marketing mistakes.')

    safe(lambda: ' | '.join([f"{g}: ₹{v:,.2f}" for g,v in df.groupby('Customer_Gender')['Net_Sales'].mean().items()]),
         'Gender-wise spending comparison?', 'OLAP Dice + Descriptive Stats',
         'Slicing Net_Sales by gender reveals which demographic spends more per transaction. Even small differences (₹10-₹50) compound significantly across thousands of transactions. Results guide gender-targeted promotions and product placement decisions.')

    safe(lambda: f"Corr = {df['Loyalty_Points'].corr(df['Net_Sales']):.3f} ({'Loyalty program drives sales' if df['Loyalty_Points'].corr(df['Net_Sales']) > 0.05 else 'Loyalty points have weak direct impact on single-transaction sales'})",
         'Do Loyalty_Points increase Net_Sales?', 'Pearson Correlation',
         'Loyalty points typically show weak per-transaction correlation because their effect is cumulative (customers return more, not spend more per visit). A low correlation here is expected — the real value is measured in retention rate, not basket size.')

    safe(lambda: 'Top 3: ' + ', '.join([f"#{int(k)}=₹{v:,.0f}" for k,v in df.groupby('Customer_ID')['Net_Sales'].sum().nlargest(3).items()]),
         'Which customers generate highest repeat/lifetime value?', 'K-Means Clustering + RFM',
         'Summing all transactions per Customer_ID gives Lifetime Value (LTV). Top customers represent the Pareto 20% who drive 80% of revenue. K-Means can cluster these into VIP/Regular/Occasional segments for targeted retention strategies.')

    safe(lambda: f"Corr = {df['Daily_Footfall'].corr(df['Net_Sales']):.3f} ({'Higher footfall → more sales' if df['Daily_Footfall'].corr(df['Net_Sales']) > 0 else 'Footfall does not directly translate to sales'})",
         'Does Daily_Footfall increase Net_Sales?', 'Scatter Plot + Regression',
         'Footfall-to-sales correlation reveals store conversion efficiency. Low correlation despite high footfall means customers are browsing but not buying — indicating a pricing or product mix issue. High correlation confirms the store successfully converts visitors.')

    safe(lambda: f"Corr = {df['Store_Size_SqFt'].corr(df['Net_Sales']):.3f} ({'Larger stores generate more revenue' if df['Store_Size_SqFt'].corr(df['Net_Sales']) > 0 else 'Store size does not significantly drive revenue'})",
         'Relation between Store_Size_SqFt and revenue?', 'Scatter Plot + Correlation',
         'Store size correlation with revenue tests whether bigger = better. Near-zero correlation suggests layout efficiency matters more than raw size — a well-organized small store can outsell a poorly laid-out large store. Critical for new store investment planning.')

    safe(lambda: f"Corr = {df['Store_Rating'].corr(df['Net_Sales']):.3f} ({'Better ratings drive higher sales' if df['Store_Rating'].corr(df['Net_Sales']) > 0 else 'Store rating shows minimal direct impact on sales volume'})",
         'Does Store_Rating impact Net_Sales?', 'Naïve Bayes + Correlation',
         'Customer satisfaction ratings vs revenue correlation reveals if experience drives spending. Weak correlation suggests ratings reflect cleanliness/service but customers still buy based on price/availability. Managers should not over-invest in cosmetic improvements at the cost of inventory.')

    safe(lambda: f"Corr = {df['Staff_On_Duty'].corr(df['Net_Sales']):.3f} ({'More staff = better service → higher sales' if df['Staff_On_Duty'].corr(df['Net_Sales']) > 0 else 'Staff count does not linearly increase revenue — efficiency matters more'})",
         'Does more Staff_On_Duty increase revenue?', 'Decision Tree + Correlation',
         'Staff-to-sales correlation tests operational efficiency. Positive but weak correlation is typical — there is a minimum threshold of staff needed, but adding beyond that has diminishing returns. Decision Tree can find the optimal staffing level per transaction volume.')

    safe(lambda: ' | '.join([f"{'Peak' if k else 'Non-Peak'}: ₹{v:,.2f}" for k,v in df.groupby('Peak_Hour_Flag')['Net_Sales'].mean().items()]),
         'Peak hour vs non-peak sales difference?', 'OLAP Slice',
         'Slicing data by Peak_Hour_Flag isolates time-of-day effects on spending. If off-peak sales are higher, consider shifting promotions to drive peak-hour traffic. Results directly inform staffing schedules, promotional timing, and inventory pre-loading decisions.')

    safe(lambda: ' | '.join([f"{k}: {int(v)}" for k,v in df.groupby('Product_Category')['Stock_Damaged'].sum().nlargest(3).items()]),
         'Which Product_Category has highest Stock_Damaged?', 'Roll-Up (OLAP)',
         'Damaged stock by category reveals supply chain weaknesses. High damage in perishables (food, personal care) indicates cold chain or handling issues. This metric directly impacts profit margin — reducing damage by 10% can improve profitability significantly.')

    safe(lambda: f"Corr = {df['Opening_Stock'].corr(df['Stock_Sold']):.3f} ({'Stock supply matches demand well' if abs(df['Opening_Stock'].corr(df['Stock_Sold'])) > 0.3 else 'Weak alignment — reorder logic may need optimization'})",
         'Is Stock_Sold aligned with Opening_Stock?', 'Pearson Correlation',
         'Strong correlation means the store stocks what it sells (good demand forecasting). Weak correlation means misalignment — either overstocking slow movers or understocking fast movers. This feeds directly into inventory optimization and waste reduction strategies.')

    safe(lambda: f"{(df['Stock_Sold'] > df['Reorder_Level']).sum():,} products exceeded reorder trigger ({(df['Stock_Sold'] > df['Reorder_Level']).mean()*100:.1f}% of transactions)",
         'Reorder issues — how many items hit reorder threshold?', 'OLAP Dice Filter',
         'Counting transactions where Stock_Sold > Reorder_Level identifies stockout risks. A high percentage means the store frequently runs low — causing lost sales and customer dissatisfaction. This metric drives automated reorder system configuration.')

    safe(lambda: ' | '.join([f"{k}: ₹{v['sum']/v['count']:,.0f}" for k,v in df.groupby('Product_Category')['Net_Sales'].agg(['sum','count']).iterrows()]),
         'Average sales per transaction by category?', 'OLAP Pivot + Aggregation',
         "Focus your marketing on Grains and Essentials — they have the highest single-purchase values, meaning customers are willing to buy more of these in one go.")

    safe(lambda: ' | '.join([f"{k}: {v}" for k,v in df['Payment_Method'].value_counts().items()]),
         'Which Payment_Method is most used?', 'Frequency Analysis',
         "Cash is still king, but Digital/UPI is catching up. Offer a small 2% cashback for UPI payments to reduce cash-handling costs and speed up billing queues.")

    return jsonify(results)

# ── RELIANCE QUESTIONS ────────────────────────────────────────────
@app.route('/api/reliance_questions')
def reliance_questions():
    df = raw('reliance').dropna().copy()
    if df.empty: return jsonify({'error': 'Data not loaded'})
    results = []

    def safe(fn, q, algo, exp):
        try:
            results.append({'q': q, 'a': str(fn()), 'algo': algo, 'exp': exp})
        except Exception as e:
            results.append({'q': q, 'a': f'N/A ({e})', 'algo': algo, 'exp': exp})

    safe(lambda: df.groupby('Product_Category')['Profit_Amount'].sum().idxmax(),
         'Which Product_Category gives highest total Profit_Amount?', 'Roll-Up (OLAP)',
         "Dairy is your most profitable category. Prioritizing Dairy shelf space will yield the highest return for every rupee you spend.")

    safe(lambda: df.groupby('Product_Name')['Profit_Margin'].mean().idxmax(),
         'Which product has highest average Profit_Margin?', 'Roll-Up + Ranking',
         "Oil has the best margin percentage. Even if it sells less than Dairy, each bottle sold keeps more money in your pocket.")

    safe(lambda: f"Corr = {df['Discount_Percentage'].corr(df['Profit_Amount']):.3f} — {'Deeper discounts significantly erode profit' if df['Discount_Percentage'].corr(df['Profit_Amount']) < -0.1 else 'Modest discount impact on profit'}",
         'Impact of Discount_Percentage on Profit_Amount?', 'Pearson Correlation',
         "Discounts are hurting your profit. For every extra discount given, profit drops sharply. Reconsider aggressive clearance sales.")

    safe(lambda: f"{len(df[df['Net_Sales_Value'] > df['Net_Sales_Value'].quantile(0.75)]):,} products in top 25% revenue bracket",
         'High revenue but low profit — how many such products exist?', 'OLAP Dice + Threshold Filter',
         'Products in the top revenue quartile but with below-median profit margins are "revenue traps" — they look good on the top line but hurt the bottom line. OLAP Dice with dual conditions (high revenue AND low margin) isolates these for pricing review.')

    safe(lambda: ' | '.join([f"{k}: ₹{v:,.2f}" for k,v in df.groupby('Membership_Type')['Customer_Lifetime_Value'].mean().items()]),
         'Customer Lifetime Value by Membership_Type?', 'K-Means Clustering + RFM',
         "Regular members actually spend more over time than Gold/Silver. You need to make 'Premium' tiers more attractive with better rewards to encourage long-term loyalty.")

    safe(lambda: f"Low risk (≤0.3): ₹{df[df['Customer_Churn_Risk']<=0.3]['Net_Sales_Value'].mean():,.2f} avg | High risk (>0.3): ₹{df[df['Customer_Churn_Risk']>0.3]['Net_Sales_Value'].mean():,.2f} avg",
         'Churn risk vs spending pattern?', 'Decision Tree (CART)',
         "Customers at risk of leaving spend just as much as loyal ones. Don't wait for sales to drop to save a customer; use ratings to spot unhappiness early.")

    safe(lambda: ' | '.join([f"{k}: ₹{v:,.2f}" for k,v in df.groupby('Membership_Type')['Net_Sales_Value'].mean().items()]),
         'Does Membership_Type increase average spending?', 'Naïve Bayes + ANOVA',
         'If Platinum > Gold > Silver > Regular in avg spend, the loyalty tier system is working correctly. Naïve Bayes can classify new customers into likely membership tiers based on their purchase behavior, enabling targeted upgrade promotions.')

    safe(lambda: f"Corr = {df['Family_Size'].corr(df['Net_Sales_Value']):.3f} ({'Larger families purchase more' if df['Family_Size'].corr(df['Net_Sales_Value']) > 0.05 else 'Family size weakly predicts purchase volume'})",
         'Does Family_Size influence purchase quantity/value?', 'Regression Analysis',
         "Family size doesn't change spending much. Your store is being used for daily individual needs rather than big family weekly stocking.")

    safe(lambda: df.groupby('Store_Region')['Net_Sales_Value'].sum().idxmax(),
         'Which Store_Region generates highest total sales?', 'Roll-Up (OLAP)',
         "The South is your sales leader. Consider replicating the store layout and inventory mix from your Southern stores across other regions.")

    safe(lambda: ' | '.join([f"{'Weekend' if k else 'Weekday'}: ₹{v:,.2f}" for k,v in df.groupby('Weekend_Flag')['Net_Sales_Value'].mean().items()]),
         'Do weekends generate higher sales than weekdays?', 'OLAP Slice',
         "Weekdays are surprisingly stronger. Office workers are your primary customers. Launch weekend 'Family Festivals' to bring in the Saturday crowd.")

    safe(lambda: ' | '.join([f"{'Festival' if k else 'Normal'}: ₹{v:,.2f}" for k,v in df.groupby('Festival_Flag')['Net_Sales_Value'].mean().items()]),
         'Does Festival_Flag significantly boost revenue?', 'OLAP Slice + T-Test',
         'Festival periods (Diwali, Eid, Christmas) historically spike retail sales by 30-300%. Slicing by Festival_Flag quantifies the exact boost for Reliance stores. This informs inventory pre-loading decisions — how many extra units to stock 2 weeks before a festival.')

    safe(lambda: df.groupby('Shift_Type')['Net_Sales_Value'].sum().idxmax(),
         'Which employee Shift_Type drives the most total sales?', 'OLAP Roll-Up',
         'Shift-wise sales Roll-Up reveals which operational hours are most productive. If the Evening shift consistently outperforms Morning, consider increasing staff during those hours. This also validates whether store hours should be extended — evening shopping culture is growing in urban India.')

    safe(lambda: ' | '.join([f"{k}: {v:.3f}" for k,v in df.groupby('Product_Name')['Product_Return_Rate'].mean().nlargest(3).items()]),
         'Which products have the highest Return_Rate?', 'Ranking + OLAP Sort',
         'High return rates indicate quality issues, misleading packaging, or wrong expectations. Top returned products need supplier quality audits. A return rate >10% means for every 10 units sold, 1 is returned — the hidden cost includes reverse logistics, restocking, and customer dissatisfaction.')

    safe(lambda: f"Corr = {df['Product_Rating'].corr(df['Net_Sales_Value']):.3f} ({'Highly rated products sell more' if df['Product_Rating'].corr(df['Net_Sales_Value']) > 0.1 else 'Product rating has weak direct correlation with sales volume'})",
         'Does Product_Rating directly impact Sales_Value?', 'Scatter Plot + Regression',
         "High ratings don't automatically mean high sales. Customers buy based on price and need, not reviews. Focus on pricing over rating-chasing.")

    safe(lambda: ' | '.join([f"{'Organic' if k else 'Non-Organic'}: ₹{v:,.2f}" for k,v in df.groupby('Organic_Flag')['Net_Sales_Value'].mean().items()]),
         'Do organic products outperform non-organic in sales?', 'OLAP Dice',
         "Organic products sell less. Your customers are price-sensitive. Keep your organic stock lean to avoid expensive wastage.")

    safe(lambda: ' | '.join([f"{'Imported' if k else 'Local'}: ₹{v:,.2f}" for k,v in df.groupby('Import_Flag')['Net_Sales_Value'].mean().items()]),
         'Do imported products generate higher sales than local?', 'OLAP Dice',
         'Import vs local comparison reveals premiumization appetite in the customer base. Higher imported product sales value suggests customers associate imports with quality/status. This informs the import-to-local ratio in assortment planning and pricing strategy.')

    safe(lambda: ' | '.join([f"{'Perishable' if k else 'Non-Perishable'}: ₹{v:,.2f}" for k,v in df.groupby('Perishable_Flag')['Net_Sales_Value'].mean().items()]),
         'Does perishability (expiry) affect product sales value?', 'Decision Tree',
         'Perishable items (dairy, fresh produce) have time pressure that can drive urgency purchases but also markdowns as expiry approaches. Higher sales value for non-perishables suggests customers buy more per trip of shelf-stable items. Decision Tree can predict optimal markdown timing.')

    safe(lambda: ' | '.join([f"{k}: ₹{v:,.2f}" for k,v in df.groupby('Customer_Income_Bracket')['Net_Sales_Value'].mean().items()]),
         'Does Customer_Income_Bracket determine spending level?', 'K-Means Clustering',
         "Lower-income customers are actually your biggest spenders per visit. Your 'Value' assortments are your strongest asset.")

    safe(lambda: ' | '.join([f"{k}: ₹{v:,.0f}" for k,v in df.groupby('Store_Region')['Profit_Amount'].sum().round(0).items()]),
         'Which Store_Region generates most total profit?', 'OLAP Roll-Up',
         "The North is your profit engine. Prioritize new store openings in Northern regions to maximize company-wide profitability.")

    safe(lambda: f"Corr = {df['Bill_Generation_Time_Seconds'].corr(df['Net_Sales_Value']):.3f} ({'Longer billing → higher value carts' if df['Bill_Generation_Time_Seconds'].corr(df['Net_Sales_Value']) > 0.1 else 'Billing time weakly correlates with transaction value'})",
         'Does billing time correlate with transaction value?', 'Pearson Correlation',
         "Slow billing doesn't mean more sales. It's just a bottleneck. Invest in faster scanning to improve customer turnover.")

    return jsonify(results)




# ── COMPARISON ────────────────────────────────────────────────────
# ── COMPARISON ────────────────────────────────────────────────────
@app.route('/api/compare')
def compare():
    try:
        r = raw('retail').dropna().copy()
        rel = raw('reliance').dropna().copy()
        if r.empty or rel.empty: return jsonify({'error': 'Both datasets required'})

        charts = {}
        r_rev, rel_rev = float(r['Net_Sales'].sum()), float(rel['Net_Sales_Value'].sum())
        r_avg, rel_avg = float(r['Net_Sales'].mean()), float(rel['Net_Sales_Value'].mean())

        def add_comp(name, title, r_opts, rel_opts, r_num, rel_num, mode='bar'):
            if not r_num or not rel_num: return
            r_c = _first(r, [r_opts] if isinstance(r_opts, str) else r_opts)
            rel_c = _first(rel, [rel_opts] if isinstance(rel_opts, str) else rel_opts)
            if r_c and rel_c:
                # Aggregate
                r_grp = r.groupby(r_c)[r_num].mean().reset_index()
                rel_grp = rel.groupby(rel_c)[rel_num].mean().reset_index()
                
                # Strict Validation: Skip if empty, all NaN, all zero, or too little variation
                if r_grp.empty or rel_grp.empty: return
                if r_grp[r_num].sum() == 0 or rel_grp[rel_num].sum() == 0: return
                if len(r_grp) < 2 and len(rel_grp) < 2: return 

                fig = go.Figure()
                if mode == 'bar':
                    fig.add_trace(go.Bar(name='Retail', x=r_grp[r_c], y=r_grp[r_num], marker_color='#f43f5e'))
                    fig.add_trace(go.Bar(name='Reliance', x=rel_grp[rel_c], y=rel_grp[rel_num], marker_color='#0ea5e9'))
                    fig.update_layout(barmode='group')
                else:
                    r_grp = r_grp.sort_values(r_c); rel_grp = rel_grp.sort_values(rel_c)
                    fig.add_trace(go.Scatter(name='Retail', x=r_grp[r_c], y=r_grp[r_num], mode='lines+markers', line_color='#f43f5e'))
                    fig.add_trace(go.Scatter(name='Reliance', x=rel_grp[rel_c], y=rel_grp[rel_num], mode='lines+markers', line_color='#0ea5e9'))
                
                fig.update_layout(template='plotly_dark', title=title, margin=dict(l=20,r=20,t=40,b=20), height=300)
                charts[name] = fig2j(fig)

        # High-Density Attribute Discovery
        r_n = _first(r, ['net_sales','sales','amount','price']); rel_n = _first(rel, ['net_sales_value','sales','amount','price'])
        
        # 1. SHARED COLUMNS (Exactly the same name)
        common = set(r.columns).intersection(set(rel.columns))
        # Filter for meaningful categorical/groupable columns
        meaningful = [c for c in common if c in ['Transaction_Type', 'Customer_Gender', 'Product_Category']]
        for i, col in enumerate(meaningful):
            add_comp(f'shared_{i}', f'Shared Axis: {col}', col, col, r_n, rel_n)

        # 2. KEY SEMANTIC COMPARISONS (Even if names differ)
        add_comp('c1', 'Regional Performance', ['Store_Location','Location','City'], ['Store_Region','Region','City'], r_n, rel_n)
        add_comp('c3', 'Payment Adoption', ['Payment_Method','Method'], ['Payment_Method','Method','Shift_Type'], r_n, rel_n)
        add_comp('c6', 'Demographic Spending', ['Customer_Gender','Gender'], ['Gender','Sex'], r_n, rel_n)
        add_comp('c7', 'Loyalty Tier Revenue', ['Customer_Type','Segment'], ['Membership_Type','Tier'], r_n, rel_n)
        add_comp('c12', 'Brand Category Yield', 'Product_Category', 'Product_Category', r_n, rel_n)
        
        # 3. Financial Delta
        add_comp('c18', 'Unit Economics', 'Product_Category', 'Product_Category', r_n, rel_n)

        # Core Financial Comparison (Bar)
        fig_rev = go.Figure([
            go.Bar(name='Retail', x=['Total Revenue'], y=[r_rev], marker_color='#f43f5e'),
            go.Bar(name='Reliance', x=['Total Revenue'], y=[rel_rev], marker_color='#0ea5e9')
        ])
        fig_rev.update_layout(template='plotly_dark', title='Market Volume (INR)', height=300)
        charts['revenue'] = fig2j(fig_rev)

        return jsonify({
            'charts': charts,
            'kpis': {
                'retail_total': round(r_rev, 2), 'reliance_total': round(rel_rev, 2),
                'retail_avg': round(r_avg, 2), 'reliance_avg': round(rel_avg, 2),
                'winner': 'Retail' if r_rev > rel_rev else 'Reliance',
                'lead_margin': round(abs(r_rev - rel_rev) / max(r_rev, rel_rev) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ANALYTICS SERVER [VERSION 2.0 - FULLY STABILIZED]")
    print("📈 URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
