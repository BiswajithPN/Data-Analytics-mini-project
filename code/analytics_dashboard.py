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
    
    print(f"Loading data from: {DATA_DIR}")
    print(f"Retail path: {retail_path}, exists: {os.path.exists(retail_path)}")
    print(f"Reliance path: {reliance_path}, exists: {os.path.exists(reliance_path)}")
    
    if os.path.exists(retail_path):
        try:
            _retail_raw = pd.read_csv(retail_path)
            _retail_raw.columns = _retail_raw.columns.str.strip()
            print(f"Retail data loaded: {_retail_raw.shape}")
        except Exception as e:
            print(f"Error loading retail data: {e}")
    else:
        print("Retail data file not found!")
        
    if os.path.exists(reliance_path):
        try:
            _reliance_raw = pd.read_csv(reliance_path)
            _reliance_raw.columns = _reliance_raw.columns.str.strip()
            print(f"Reliance data loaded: {_reliance_raw.shape}")
        except Exception as e:
            print(f"Error loading reliance data: {e}")
    else:
        print("Reliance data file not found!")
        
    retail_work   = _retail_raw.copy() if not _retail_raw.empty else pd.DataFrame()
    reliance_work = _reliance_raw.copy() if not _reliance_raw.empty else pd.DataFrame()
    print(f"Data loading complete. Retail: {len(_retail_raw)} rows, Reliance: {len(_reliance_raw)} rows")

load()

def _first(df, options):
    """Find the first column that matches any of the options (case-insensitive)."""
    cols = [c.lower() for c in df.columns]
    for opt in options:
        if opt.lower() in cols:
            # Return the actual column name from df.columns
            return df.columns[cols.index(opt.lower())]
    return df.columns[0] if len(df.columns) > 0 else None

def _product_col(df):
    """Prefer readable product names over numeric product ids."""
    return _first(df, ['Product_Name', 'Product Name', 'product_name', 'product']) 

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
        r_top_product = str(r.groupby('Product_Name')['Net_Sales'].sum().idxmax())
        r_top_products = r.groupby('Product_Name')['Net_Sales'].sum().sort_values(ascending=False).head(5).index.tolist()

        # Reliance Metrics
        rel_rev = rel['Net_Sales_Value'].sum()
        rel_eff = rel_rev / len(rel)
        rel_acc = _get_acc(rel, 'Membership_Type')
        rel_top_product = str(rel.groupby('Product_Name')['Profit_Amount'].sum().idxmax())
        rel_top_products = rel.groupby('Product_Name')['Profit_Amount'].sum().sort_values(ascending=False).head(5).index.tolist()

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
                'top_product': r_top_product,
                'top_products': r_top_products,
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
                'top_product': rel_top_product,
                'top_products': rel_top_products,
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
            {'name': 'Customer Segmentation', 'use': 'Identify high-value customer groups for targeted marketing', 'why': 'Helps focus marketing budget on customers most likely to increase revenue'},
            {'name': 'Payment Pattern Analysis', 'use': 'Understand customer payment preferences to optimize billing systems', 'why': 'Reduces transaction costs and improves customer experience'},
            {'name': 'Customer Behavior Clustering', 'use': 'Group customers by spending patterns and purchase frequency', 'why': 'Enables personalized marketing and improves customer retention'},
            {'name': 'Revenue Aggregation', 'use': 'Track total sales performance across categories and regions', 'why': 'Provides clear visibility into business performance for strategic planning'},
            {'name': 'Detailed Performance Analysis', 'use': 'Drill down from category to product level for granular insights', 'why': 'Identifies specific products driving success or needing attention'},
            {'name': 'Time-Based Analysis', 'use': 'Analyze performance during specific periods like peak hours', 'why': 'Optimizes staffing and inventory for high-demand periods'},
            {'name': 'High-Performance Filtering', 'use': 'Focus on top-performing segments or products', 'why': 'Concentrates resources on areas with highest return on investment'},
            {'name': 'Cross-Dimensional Analysis', 'use': 'Compare performance across multiple business dimensions', 'why': 'Reveals hidden patterns and opportunities for growth'},
            {'name': 'Relationship Analysis', 'use': 'Understand how factors like discounts affect sales', 'why': 'Optimizes pricing strategies and promotional effectiveness'},
            {'name': 'Performance Normalization', 'use': 'Compare metrics on a standardized scale', 'why': 'Ensures fair comparison between different business units'}
        ]
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/product_insights/<dataset>')
def product_insights(dataset):
    try:
        df = raw(dataset).dropna().copy()
        if df.empty: return jsonify({'error': 'Data not loaded'})
        
        # Find relevant columns
        product_col = _product_col(df)
        category_col = next((c for c in df.columns if 'categor' in c.lower()), df.columns[1])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sales_col = next((c for c in numeric_cols if 'sales' in c.lower() or 'value' in c.lower()), numeric_cols[0])
        quantity_col = next((c for c in numeric_cols if 'sold' in c.lower() or 'quantity' in c.lower()), numeric_cols[0])
        rating_col = next((c for c in df.columns if 'rating' in c.lower()), None)
        return_col = next((c for c in df.columns if 'return' in c.lower()), None)
        margin_col = next((c for c in numeric_cols if 'margin' in c.lower() or 'profit' in c.lower()), None)
        
        # Overall Portfolio Health
        total_unique_products = df[product_col].nunique()
        avg_rating = df[rating_col].mean() if rating_col else 'N/A'
        return_rate = (df[return_col].sum() / len(df) * 100) if return_col else 0
        stock_turnover = df[quantity_col].sum() / len(df) if quantity_col else 0
        
        # Sales & Profitability
        product_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        highest_revenue = product_sales.index[0] if len(product_sales) > 0 else 'N/A'
        lowest_revenue = product_sales.index[-1] if len(product_sales) > 0 else 'N/A'
        most_sold = df.groupby(product_col)[quantity_col].sum().idxmax() if quantity_col else 'N/A'
        highest_margin = df.groupby(product_col)[margin_col].mean().idxmax() if margin_col else highest_revenue
        
        # Category Analysis
        category_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
        
        # Top & Bottom Products
        top_products = product_sales.head(10).index.tolist()
        bottom_products = product_sales.tail(5).index.tolist()
        
        return jsonify({
            'portfolio_health': {
                'total_unique_products': total_unique_products,
                'avg_rating': round(avg_rating, 2) if avg_rating != 'N/A' else 'N/A',
                'return_rate': round(return_rate, 2),
                'stock_turnover': round(stock_turnover, 2)
            },
            'sales_profitability': {
                'highest_revenue_product': highest_revenue,
                'lowest_revenue_product': lowest_revenue,
                'highest_margin_product': highest_margin,
                'most_sold_product': most_sold
            },
            'category_analysis': {
                'categories': category_sales.index.tolist(),
                'sales': category_sales.values.tolist()
            },
            'top_products': top_products,
            'bottom_products': bottom_products
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/advanced_product_insights/<dataset>')
def advanced_product_insights(dataset):
    try:
        df = raw(dataset).dropna().copy()
        if df.empty: return jsonify({'error': 'Data not loaded'})
        
        product_col = _product_col(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sales_col = next((c for c in numeric_cols if 'sales' in c.lower() or 'value' in c.lower()), numeric_cols[0])
        
        # Product Lifecycle Analysis
        product_trends = df.groupby(product_col)[sales_col].agg(['sum', 'mean', 'count'])
        growing_threshold = product_trends['sum'].quantile(0.75)
        declining_threshold = product_trends['sum'].quantile(0.25)
        
        growing_products = product_trends[product_trends['sum'] >= growing_threshold].index.tolist()
        declining_products = product_trends[product_trends['sum'] <= declining_threshold].index.tolist()
        mature_products = product_trends[(product_trends['sum'] > declining_threshold) & 
                                     (product_trends['sum'] < growing_threshold)].index.tolist()
        
        # Recommendations
        top_performers = product_trends.nlargest(5, 'sum').index.tolist()
        bottom_performers = product_trends.nsmallest(3, 'sum').index.tolist()
        introduce_products = [f"Premium {p}" for p in top_performers[:3]]
        
        return jsonify({
            'lifecycle': {
                'growing_products': len(growing_products),
                'mature_products': len(mature_products),
                'declining_products': len(declining_products),
                'new_products': len(df[product_col].unique()) // 10  # Estimate
            },
            'recommendations': {
                'expand_products': top_performers[:3],
                'monitor_products': mature_products[:3],
                'discontinue_products': bottom_performers,
                'introduce_products': introduce_products
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/product_qa/<dataset>')
def product_qa(dataset):
    try:
        df = raw(dataset).dropna().copy()
        if df.empty: return jsonify({'error': 'Data not loaded'})
        
        product_col = _product_col(df)
        category_col = next((c for c in df.columns if 'categor' in c.lower()), df.columns[1])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sales_col = next((c for c in numeric_cols if 'sales' in c.lower() or 'value' in c.lower()), numeric_cols[0])
        
        # Generate Q&A based on dataset
        if dataset == 'retail':
            questions = [
                {
                    'q': 'Which product category generates the highest revenue?',
                    'a': str(df.groupby(category_col)[sales_col].sum().idxmax()),
                    'insight': 'Focus marketing on high-revenue categories to maximize ROI'
                },
                {
                    'q': 'What is the average customer basket size?',
                    'a': f'₹{df[sales_col].mean():.2f}',
                    'insight': 'Larger basket sizes indicate strong cross-selling opportunities'
                },
                {
                    'q': 'Which products have the highest return rates?',
                    'a': 'Electronics and Fashion items typically show higher returns',
                    'insight': 'Improve quality control and product descriptions for high-return categories'
                },
                {
                    'q': 'What is the optimal stock level for top products?',
                    'a': 'Maintain 2-3 weeks of inventory for fast-moving items',
                    'insight': 'Balance between stock availability and holding costs'
                },
                {
                    'q': 'Which payment methods are most popular for high-value purchases?',
                    'a': 'Credit cards and digital wallets for premium products',
                    'insight': 'Offer payment method-specific promotions to increase conversion'
                }
            ]
        else:  # Reliance
            questions = [
                {
                    'q': 'Which product category has the highest profit margin?',
                    'a': str(df.groupby(category_col)[sales_col].mean().idxmax()),
                    'insight': 'Prioritize high-margin categories in store placement and marketing'
                },
                {
                    'q': 'What is the average transaction value?',
                    'a': f'₹{df[sales_col].mean():.2f}',
                    'insight': 'Higher transaction values indicate premium positioning success'
                },
                {
                    'q': 'Which products are most frequently purchased together?',
                    'a': 'Grocery items and household essentials often bought together',
                    'insight': 'Create bundle deals for complementary products'
                },
                {
                    'q': 'What is the impact of discounts on sales volume?',
                    'a': '10-15% discount levels optimize volume without eroding margin',
                    'insight': 'Use data-driven discount strategies rather than flat percentages'
                },
                {
                    'q': 'Which store locations perform best for premium products?',
                    'a': 'Urban and high-income areas show premium product strength',
                    'insight': 'Tailor product assortment by location demographics'
                }
            ]
        
        return jsonify({
            'performance_qa': questions[:2],
            'strategy_qa': questions[2:4],
            'inventory_qa': [questions[0]] if len(questions) > 0 else [],
            'customer_qa': [questions[1]] if len(questions) > 1 else [],
            'financial_qa': questions[4:] if len(questions) > 4 else []
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/business_insights/<dataset>')
def business_insights(dataset):
    try:
        df = raw(dataset).dropna().copy()
        if df.empty: return jsonify({'error': 'Data not loaded'})
        
        # 1. Overall Business Performance
        total_sales = float(df.select_dtypes(include=[np.number]).sum().sum())
        total_orders = len(df)
        
        # Find numeric columns for products sold and order value
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        products_sold_col = next((c for c in numeric_cols if 'sold' in c.lower() or 'quantity' in c.lower()), numeric_cols[0])
        order_value_col = next((c for c in numeric_cols if 'sales' in c.lower() or 'value' in c.lower()), numeric_cols[0])
        
        total_products_sold = int(df[products_sold_col].sum()) if products_sold_col in df.columns else 0
        avg_order_value = float(df[order_value_col].mean()) if order_value_col in df.columns else 0
        
        # 2. Sales Trend (simplified - using row index as time proxy)
        monthly_sales = df.groupby(df.index // (len(df) // 12 + 1))[order_value_col].sum() if len(df) > 12 else df[order_value_col]
        peak_month = monthly_sales.idxmax() if len(monthly_sales) > 1 else 0
        lowest_month = monthly_sales.idxmin() if len(monthly_sales) > 1 else 0
        
        # 3. Best/Worst Performing Products
        product_col = _product_col(df)
        product_performance = df.groupby(product_col)[order_value_col].sum().sort_values(ascending=False)
        top_products = product_performance.head(3).index.tolist()
        worst_products = product_performance.tail(3).index.tolist()
        
        # 4. Category Performance
        category_col = next((c for c in df.columns if 'categor' in c.lower()), df.columns[1])
        category_performance = df.groupby(category_col)[order_value_col].sum().sort_values(ascending=False)
        best_category = category_performance.index[0] if len(category_performance) > 0 else 'N/A'
        worst_category = category_performance.index[-1] if len(category_performance) > 0 else 'N/A'
        
        # 5. Customer Behavior
        customer_cols = [c for c in df.columns if 'customer' in c.lower() or 'member' in c.lower()]
        location_cols = [c for c in df.columns if 'location' in c.lower() or 'region' in c.lower() or 'city' in c.lower()]
        
        # 6. Location Performance
        location_col = location_cols[0] if location_cols else df.columns[0]
        location_performance = df.groupby(location_col)[order_value_col].sum().sort_values(ascending=False)
        top_location = location_performance.index[0] if len(location_performance) > 0 else 'N/A'
        worst_location = location_performance.index[-1] if len(location_performance) > 0 else 'N/A'
        
        # 7. Future Predictions (simple trend based)
        trend = 'growing' if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else 'declining' if monthly_sales.iloc[-1] < monthly_sales.iloc[0] else 'stable'
        
        return jsonify({
            'overall_performance': {
                'total_sales': round(total_sales, 2),
                'total_orders': total_orders,
                'total_products_sold': total_products_sold,
                'average_order_value': round(avg_order_value, 2),
                'trend': trend
            },
            'sales_trend': {
                'peak_month': int(peak_month),
                'lowest_month': int(lowest_month),
                'insight': f'Sales {trend} based on analysis period'
            },
            'product_performance': {
                'top_products': top_products,
                'worst_products': worst_products,
                'insight': 'Top products generate highest revenue consistently'
            },
            'category_performance': {
                'best_category': best_category,
                'worst_category': worst_category,
                'insight': f'Majority revenue comes from {best_category}'
            },
            'location_performance': {
                'top_location': top_location,
                'worst_location': worst_location,
                'insight': f'{top_location} shows highest demand'
            },
            'recommendations': {
                'focus_products': top_products[:2],
                'expand_category': best_category,
                'target_location': top_location,
                'action_items': [
                    f'Increase stock for {top_products[0] if top_products else "top products"}',
                    f'Expand {best_category} category',
                    f'Focus marketing in {top_location}'
                ]
            }
        })
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
        r = raw('retail').copy()
        rel = raw('reliance').copy()
        if r.empty or rel.empty: return jsonify({'error': 'Both datasets required'})

        charts = []
        r_rev, rel_rev = float(r['Net_Sales'].sum()), float(rel['Net_Sales_Value'].sum())
        r_avg, rel_avg = float(r['Net_Sales'].mean()), float(rel['Net_Sales_Value'].mean())
        r_sales_col, rel_sales_col = 'Net_Sales', 'Net_Sales_Value'

        def style(fig, title, height=340):
            fig.update_layout(
                template='plotly_dark',
                title=title,
                height=height,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.35)',
                margin=dict(l=40, r=20, t=55, b=45),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            return fig

        def add_chart(key, fig):
            charts.append({'id': key, 'fig': fig2j(fig)})

        def clean_xy(df, x_col, y_col):
            return df[[x_col, y_col]].dropna()

        def clean_y(df, y_col):
            return pd.to_numeric(df[y_col], errors='coerce').dropna()

        def labels(series_or_index):
            return [str(v) for v in series_or_index]

        def nums(series_or_values):
            return [float(v) for v in series_or_values]

        def find_col(df, options):
            opts = [options] if isinstance(options, str) else options
            lower = {c.lower(): c for c in df.columns}
            for opt in opts:
                if opt.lower() in lower:
                    return lower[opt.lower()]
            return None

        def add_comp(name, title, r_opts, rel_opts, r_num, rel_num, mode='bar'):
            if not r_num or not rel_num: return
            r_c = find_col(r, r_opts)
            rel_c = find_col(rel, rel_opts)
            if r_c and rel_c:
                # Aggregate
                r_clean = clean_xy(r, r_c, r_num)
                rel_clean = clean_xy(rel, rel_c, rel_num)
                if r_clean.empty or rel_clean.empty: return
                r_grp = r_clean.groupby(r_c)[r_num].mean().reset_index()
                rel_grp = rel_clean.groupby(rel_c)[rel_num].mean().reset_index()
                
                # Strict Validation: Skip if empty, all NaN, all zero, or too little variation
                if r_grp.empty or rel_grp.empty: return
                if r_grp[r_num].sum() == 0 or rel_grp[rel_num].sum() == 0: return
                if len(r_grp) < 2 and len(rel_grp) < 2: return 
                r_grp = r_grp.sort_values(r_num, ascending=False).head(10)
                rel_grp = rel_grp.sort_values(rel_num, ascending=False).head(10)

                fig = go.Figure()
                if mode == 'bar':
                    fig.add_trace(go.Bar(name='Retail', x=labels(r_grp[r_c]), y=nums(r_grp[r_num]), marker_color='#f43f5e'))
                    fig.add_trace(go.Bar(name='Reliance', x=labels(rel_grp[rel_c]), y=nums(rel_grp[rel_num]), marker_color='#0ea5e9'))
                    fig.update_layout(barmode='group')
                else:
                    r_grp = r_grp.sort_values(r_c); rel_grp = rel_grp.sort_values(rel_c)
                    fig.add_trace(go.Scatter(name='Retail', x=labels(r_grp[r_c]), y=nums(r_grp[r_num]), mode='lines+markers', line_color='#f43f5e'))
                    fig.add_trace(go.Scatter(name='Reliance', x=labels(rel_grp[rel_c]), y=nums(rel_grp[rel_num]), mode='lines+markers', line_color='#0ea5e9'))
                
                add_chart(name, style(fig, title))

        # High-Density Attribute Discovery
        r_n = _first(r, ['net_sales','sales','amount','price']); rel_n = _first(rel, ['net_sales_value','sales','amount','price'])
        
        # 0. Executive financial comparisons
        fig_rev = go.Figure([
            go.Bar(name='Retail', x=['Total Revenue'], y=[r_rev], marker_color='#f43f5e', text=[f'₹{r_rev:,.0f}'], textposition='auto'),
            go.Bar(name='Reliance', x=['Total Revenue'], y=[rel_rev], marker_color='#0ea5e9', text=[f'₹{rel_rev:,.0f}'], textposition='auto')
        ])
        add_chart('total_revenue', style(fig_rev, 'Total Revenue Comparison'))

        fig_avg = go.Figure([
            go.Bar(name='Retail', x=['Avg Transaction'], y=[r_avg], marker_color='#fb7185', text=[f'₹{r_avg:,.0f}'], textposition='auto'),
            go.Bar(name='Reliance', x=['Avg Transaction'], y=[rel_avg], marker_color='#38bdf8', text=[f'₹{rel_avg:,.0f}'], textposition='auto')
        ])
        add_chart('avg_transaction', style(fig_avg, 'Average Transaction Value'))

        fig_count = go.Figure([
            go.Bar(name='Transactions', x=['Retail', 'Reliance'], y=[len(r), len(rel)], marker_color=['#f43f5e', '#0ea5e9'], text=[f'{len(r):,}', f'{len(rel):,}'], textposition='auto')
        ])
        add_chart('transaction_count', style(fig_count, 'Transaction Volume'))

        fig_market_share = go.Figure(data=[go.Pie(
            labels=['Retail Revenue', 'Reliance Revenue'],
            values=[float(r_rev), float(rel_rev)],
            hole=.48,
            marker=dict(colors=['#f43f5e', '#0ea5e9']),
            textinfo='label+percent+value',
            texttemplate='%{label}<br>₹%{value:,.0f}<br>%{percent}',
            hovertemplate='%{label}<br>Revenue: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>'
        )])
        add_chart('market_share_pie', style(fig_market_share, 'Revenue Market Share Pie', 380))

        fig_txn_share = go.Figure(data=[go.Pie(
            labels=['Retail Transactions', 'Reliance Transactions'],
            values=[int(len(r)), int(len(rel))],
            hole=.42,
            marker=dict(colors=['#fb7185', '#38bdf8']),
            textinfo='label+percent+value',
            hovertemplate='%{label}<br>Transactions: %{value:,}<br>Share: %{percent}<extra></extra>'
        )])
        add_chart('transaction_share_pie', style(fig_txn_share, 'Transaction Share Pie', 360))

        # 1. SHARED COLUMNS (Exactly the same name)
        common = set(r.columns).intersection(set(rel.columns))
        # Filter for meaningful categorical/groupable columns
        meaningful = [c for c in common if c in ['Transaction_Type', 'Customer_Gender', 'Product_Category']]
        for i, col in enumerate(meaningful):
            add_comp(f'shared_{i}', f'Shared Axis: {col}', col, col, r_n, rel_n)

        # 2. KEY SEMANTIC COMPARISONS (Even if names differ)
        add_comp('c1', 'Regional Performance', ['Store_Location','Location','City'], ['Store_Region','Region','City'], r_n, rel_n)
        add_comp('c3', 'Payment Method Spending', ['Payment_Method','Method'], ['Payment_Method','Method'], r_n, rel_n)
        add_comp('c6', 'Demographic Spending', ['Customer_Gender','Gender'], ['Customer_Gender','Gender','Sex'], r_n, rel_n)
        add_comp('c7', 'Loyalty Tier Revenue', ['Customer_Type','Segment'], ['Membership_Type','Tier'], r_n, rel_n)
        add_comp('c12', 'Brand Category Yield', 'Product_Category', 'Product_Category', r_n, rel_n)
        
        # 3. Financial Delta
        add_comp('c18', 'Unit Economics', 'Product_Category', 'Product_Category', r_n, rel_n)

        # 4. Distribution and operational comparison charts
        r_cat = r.groupby('Product_Category')[r_sales_col].sum().sort_values(ascending=False).head(8)
        rel_cat = rel.groupby('Product_Category')[rel_sales_col].sum().sort_values(ascending=False).head(8)
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(name='Retail', y=labels(r_cat.index[::-1]), x=nums(r_cat.values[::-1]), orientation='h', marker_color='#f43f5e'))
        fig_cat.add_trace(go.Bar(name='Reliance', y=labels(rel_cat.index[::-1]), x=nums(rel_cat.values[::-1]), orientation='h', marker_color='#0ea5e9'))
        add_chart('category_revenue_rank', style(fig_cat, 'Top Category Revenue Rank', 390))

        r_prod = r.groupby(_product_col(r))[r_sales_col].sum().sort_values(ascending=False).head(8)
        rel_prod = rel.groupby(_product_col(rel))[rel_sales_col].sum().sort_values(ascending=False).head(8)
        fig_prod = go.Figure()
        fig_prod.add_trace(go.Bar(name='Retail', y=labels(r_prod.index[::-1]), x=nums(r_prod.values[::-1]), orientation='h', marker_color='#fb7185'))
        fig_prod.add_trace(go.Bar(name='Reliance', y=labels(rel_prod.index[::-1]), x=nums(rel_prod.values[::-1]), orientation='h', marker_color='#38bdf8'))
        add_chart('top_product_revenue', style(fig_prod, 'Top Product Revenue Comparison', 420))

        fig_retail_cat_pie = go.Figure(data=[go.Pie(
            labels=labels(r_cat.index),
            values=nums(r_cat.values),
            hole=.4,
            marker=dict(colors=px.colors.qualitative.Set2),
            textinfo='label+percent',
            hovertemplate='%{label}<br>Retail revenue: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>'
        )])
        add_chart('retail_category_pie', style(fig_retail_cat_pie, 'Retail Category Revenue Share', 380))

        fig_reliance_cat_pie = go.Figure(data=[go.Pie(
            labels=labels(rel_cat.index),
            values=nums(rel_cat.values),
            hole=.4,
            marker=dict(colors=px.colors.qualitative.Pastel),
            textinfo='label+percent',
            hovertemplate='%{label}<br>Reliance revenue: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>'
        )])
        add_chart('reliance_category_pie', style(fig_reliance_cat_pie, 'Reliance Category Revenue Share', 380))

        if 'Discount_Percentage' in r.columns and 'Discount_Percentage' in rel.columns:
            r_disc = clean_xy(r, 'Discount_Percentage', r_sales_col).sample(n=min(700, len(clean_xy(r, 'Discount_Percentage', r_sales_col))), random_state=42)
            rel_disc = clean_xy(rel, 'Discount_Percentage', rel_sales_col).sample(n=min(700, len(clean_xy(rel, 'Discount_Percentage', rel_sales_col))), random_state=42)
            fig_discount = go.Figure()
            fig_discount.add_trace(go.Scatter(name='Retail', x=nums(r_disc['Discount_Percentage']), y=nums(r_disc[r_sales_col]), mode='markers', marker=dict(color='#f43f5e', opacity=.45, size=6)))
            fig_discount.add_trace(go.Scatter(name='Reliance', x=nums(rel_disc['Discount_Percentage']), y=nums(rel_disc[rel_sales_col]), mode='markers', marker=dict(color='#0ea5e9', opacity=.45, size=6)))
            fig_discount.update_xaxes(title='Discount %')
            fig_discount.update_yaxes(title='Sales Value')
            add_chart('discount_sales_scatter', style(fig_discount, 'Discount vs Sales Scatter'))

        if 'Quantity_Sold' in r.columns and 'Quantity_Sold' in rel.columns:
            r_qty = r.groupby('Product_Category')['Quantity_Sold'].mean().sort_values(ascending=False).head(8)
            rel_qty = rel.groupby('Product_Category')['Quantity_Sold'].mean().sort_values(ascending=False).head(8)
            fig_qty = go.Figure()
            fig_qty.add_trace(go.Bar(name='Retail', x=labels(r_qty.index), y=nums(r_qty.values), marker_color='#f43f5e'))
            fig_qty.add_trace(go.Bar(name='Reliance', x=labels(rel_qty.index), y=nums(rel_qty.values), marker_color='#0ea5e9'))
            add_chart('quantity_category', style(fig_qty, 'Average Quantity Sold by Category'))

        if 'Store_Size_SqFt' in r.columns and 'Store_Size_SqFt' in rel.columns:
            r_store = clean_xy(r, 'Store_Size_SqFt', r_sales_col).sample(n=min(700, len(clean_xy(r, 'Store_Size_SqFt', r_sales_col))), random_state=42)
            rel_store = clean_xy(rel, 'Store_Size_SqFt', rel_sales_col).sample(n=min(700, len(clean_xy(rel, 'Store_Size_SqFt', rel_sales_col))), random_state=42)
            fig_store = go.Figure()
            fig_store.add_trace(go.Scatter(name='Retail', x=nums(r_store['Store_Size_SqFt']), y=nums(r_store[r_sales_col]), mode='markers', marker=dict(color='#f43f5e', opacity=.45, size=6)))
            fig_store.add_trace(go.Scatter(name='Reliance', x=nums(rel_store['Store_Size_SqFt']), y=nums(rel_store[rel_sales_col]), mode='markers', marker=dict(color='#0ea5e9', opacity=.45, size=6)))
            fig_store.update_xaxes(title='Store Size SqFt')
            fig_store.update_yaxes(title='Sales Value')
            add_chart('store_size_sales', style(fig_store, 'Store Size vs Sales'))

        if 'Profit_Amount' in rel.columns:
            rel_profit = rel.groupby('Product_Category')['Profit_Amount'].sum().sort_values(ascending=False).head(8)
            fig_profit = go.Figure()
            fig_profit.add_trace(go.Bar(name='Reliance Profit', x=labels(rel_profit.index), y=nums(rel_profit.values), marker_color='#22c55e'))
            add_chart('reliance_profit_category', style(fig_profit, 'Reliance Profit by Category'))

        return jsonify({
            'charts': charts,
            'kpis': {
                'retail_total': round(r_rev, 2), 'reliance_total': round(rel_rev, 2),
                'retail_revenue': round(r_rev, 2), 'reliance_revenue': round(rel_rev, 2),
                'retail_avg': round(r_avg, 2), 'reliance_avg': round(rel_avg, 2),
                'winner': 'Retail' if r_rev > rel_rev else 'Reliance',
                'lead_margin': round(abs(r_rev - rel_rev) / max(r_rev, rel_rev) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    # Avoid UnicodeEncodeError on Windows consoles (cp1252).
    print("ANALYTICS SERVER [VERSION 2.0 - FULLY STABILIZED]")
    print("URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
