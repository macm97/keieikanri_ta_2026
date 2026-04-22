"""
経営管理論 2026 中間プロジェクト 模範解答
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import unicodedata
import datetime as dt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

DATA_PATH = 'data_j_2026.xls'
PERIOD_DAYS = 252  # 1年 = 252 営業日


# ─────────────────────────────────────────────
# ユーティリティ関数
# ─────────────────────────────────────────────

def normalize_str(s):
    """全角英数字・記号を半角に変換"""
    return unicodedata.normalize('NFKC', s) if isinstance(s, str) else s


def load_company_list(path):
    """銘柄一覧を読み込み（ETF・ETN除外、全角→半角正規化）"""
    df = pd.read_excel(path)
    df = df.replace('-', np.nan)
    df['銘柄名'] = df['銘柄名'].apply(normalize_str)
    df = df[df['33業種コード'].notna()].reset_index(drop=True)
    # 2022年以降の英数字コード（例: 130A）に対応するため文字列として保持
    df['コード'] = df['コード'].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    df['コード&銘柄名'] = df['コード'] + ' ' + df['銘柄名']
    return df


def get_tickers(df_all, selections):
    """選択銘柄から Yahoo Finance ティッカー（1301.T 形式）と表示名を取得"""
    df_sel = df_all[df_all['コード&銘柄名'].isin(selections)].copy()
    tickers = [str(c) + '.T' for c in df_sel['コード']]
    stock_names = sorted(selections)  # ソート済みで列名に使う
    return df_sel, tickers, stock_names


def fetch_price_and_returns(tickers, stock_names, start, end):
    """Yahoo Finance から株価を取得し、価格 DataFrame と対数収益率 DataFrame を返す"""
    dfs_price, dfs_ret = [], []
    for ticker, name in zip(tickers, stock_names):
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            raise ValueError(
                f"{ticker}: データが見つかりませんでした。"
                "銘柄コードまたは期間を確認してください。"
            )
        # タイムゾーンを除去して日付のみに正規化
        hist.index = pd.to_datetime(hist.index.date)
        close = hist['Close'].rename(name)
        price_s = close.reset_index()
        price_s.columns = ['Date', name]
        # 昇順データなので shift(1) で前日の価格を取得
        log_ret_s = (np.log(close) - np.log(close.shift(1))).dropna()
        log_ret_df = pd.DataFrame({'Date': log_ret_s.index, name: log_ret_s.values})
        dfs_price.append(price_s)
        dfs_ret.append(log_ret_df)

    df_p = dfs_price[0]
    df_r = dfs_ret[0]
    for i in range(1, len(tickers)):
        df_p = pd.merge(df_p, dfs_price[i], on='Date')
        df_r = pd.merge(df_r, dfs_ret[i], on='Date')

    df_p['Date'] = pd.to_datetime(df_p['Date']).dt.normalize()
    df_r['Date'] = pd.to_datetime(df_r['Date']).dt.normalize()
    return df_p, df_r


def run_monte_carlo(df_nd, N, period, zero_corr=False):
    """
    モンテカルロ法でポートフォリオの (収益率, 標準偏差) を計算する。
    末尾 n 行は個別銘柄（投資比率 100%）。
    """
    n = df_nd.shape[1]
    vcm = df_nd.cov().values
    if zero_corr:
        vcm = np.diag(np.diag(vcm))  # 対角成分のみ残す（共分散 = 0）

    np_vcm = vcm * period               # 1年分分散共分散行列
    np_mean = df_nd.mean().values * period  # 1年分期待収益率ベクトル

    weights = np.random.uniform(size=(N, n))
    weights /= weights.sum(axis=1, keepdims=True)
    weights = np.vstack([weights, np.identity(n)])  # 末尾に個別銘柄を追加

    rows = []
    for w in weights:
        rp = w @ np_mean
        vp = w @ np_vcm @ w
        rows.append((w.tolist(), rp, float(vp)))

    df2 = pd.DataFrame(rows, columns=['投資比率', '収益率', '収益率の分散'])
    df2['収益率の標準偏差'] = np.sqrt(df2['収益率の分散'].clip(lower=0))
    return df2


def find_tangent_portfolio(df_mc, n_stocks, rf_period):
    """
    MC 結果から接点ポートフォリオ（シャープレシオ最大）を探す。
    個別銘柄行（末尾 n 行）は除いて探索。
    """
    df_pf = df_mc.iloc[:-n_stocks].copy()
    df_pf = df_pf[df_pf['収益率の標準偏差'] > 1e-8]
    sr = (df_pf['収益率'] - rf_period) / df_pf['収益率の標準偏差']
    t_idx = sr.idxmax()
    return df_mc.at[t_idx, '収益率の標準偏差'], df_mc.at[t_idx, '収益率'], sr[t_idx]


def make_frontier_figure(df_mc, stock_names, rf_period, rf_pct, title):
    """有効フロンティア散布図 + CML + 接点ポートフォリオを描画する"""
    n = len(stock_names)
    df_pf = df_mc.iloc[:-n]
    df_stocks = df_mc.iloc[-n:].reset_index(drop=True)

    t_sigma, t_mu, t_sr = find_tangent_portfolio(df_mc, n, rf_period)

    sigma_max = df_mc['収益率の標準偏差'].max() * 1.1
    sig_cml = np.linspace(0, sigma_max, 300)
    mu_cml = rf_period + t_sr * sig_cml

    fig = go.Figure()

    # MC ポートフォリオ群
    fig.add_trace(go.Scatter(
        x=df_pf['収益率の標準偏差'],
        y=df_pf['収益率'],
        mode='markers',
        marker=dict(size=3, color='steelblue', opacity=0.35),
        name='MC ポートフォリオ',
        hovertemplate='σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>',
    ))

    # 個別銘柄
    colors = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    for i, name in enumerate(stock_names):
        fig.add_trace(go.Scatter(
            x=[df_stocks.at[i, '収益率の標準偏差']],
            y=[df_stocks.at[i, '収益率']],
            mode='markers+text',
            text=[name],
            textposition='top right',
            marker=dict(size=13, color=colors[i % len(colors)],
                        line=dict(color='black', width=1)),
            name=name,
        ))

    # 無リスク資産
    fig.add_trace(go.Scatter(
        x=[0], y=[rf_period],
        mode='markers+text',
        text=[f'  rf={rf_pct:.2f}%'],
        textposition='middle right',
        marker=dict(size=12, color='black', symbol='diamond'),
        name=f'無リスク資産 (rf={rf_pct:.2f}%)',
    ))

    # 資本市場線（CML）
    fig.add_trace(go.Scatter(
        x=sig_cml, y=mu_cml,
        mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'資本市場線 CML (傾き={t_sr:.3f})',
    ))

    # 接点ポートフォリオ
    fig.add_trace(go.Scatter(
        x=[t_sigma], y=[t_mu],
        mode='markers+text',
        text=[f'  接点PF (SR={t_sr:.3f})'],
        textposition='middle right',
        marker=dict(size=18, color='gold', symbol='star',
                    line=dict(color='black', width=1)),
        name=f'接点ポートフォリオ (SR={t_sr:.3f})',
    ))

    fig.update_layout(
        title=title,
        xaxis_title='収益率の標準偏差 σ（1年）',
        yaxis_title='期待収益率 μ（1年）',
        height=560, width=920,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    )
    return fig, t_sigma, t_mu, t_sr


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    st.title('経営管理論 中間プロジェクト 模範解答 2026')
    st.write('3 銘柄を選択 → パラメータ設定 → Submit で模範解答を生成します')

    # ── 銘柄選択 ──────────────────────────────
    st.header('銘柄選択')

    df_all = load_company_list(DATA_PATH)

    with st.expander(f'全銘柄一覧（ETF・ETN 除く、{len(df_all)} 銘柄）を表示'):
        st.dataframe(df_all, use_container_width=True)

    selections_temp = st.multiselect(
        '銘柄を選択してください（3 銘柄を想定）',
        df_all['コード&銘柄名'],
    )
    selections = sorted(selections_temp)

    if selections:
        df_sel, tickers, stock_names = get_tickers(df_all, selections)
        st.write('選択銘柄:')
        st.dataframe(
            df_sel[['コード', '銘柄名', '市場・商品区分', '33業種区分']],
            use_container_width=True,
        )

    with st.expander('for developer'):
        if selections:
            st.write('stock_names:', stock_names)
            st.write('tickers:', tickers)

    # ── パラメータ設定 ─────────────────────────
    st.header('パラメータ設定')
    col1, col2, col3 = st.columns(3)

    with col1:
        date_start = st.date_input('データ開始日', dt.date(2025, 4, 1))
        date_end   = st.date_input('データ終了日',   dt.date(2026, 4, 1))

    with col2:
        N = st.slider('モンテカルロ 試行回数', 10000, 50000, 20000, step=5000)

    with col3:
        rf_pct = st.number_input(
            f'無リスク利子率（年率・%）\n日本国債 30 年利回りを入力',
            min_value=0.0, max_value=10.0,
            value=2.50, step=0.05, format='%.2f',
        )
        rf_annual = rf_pct / 100.0
        # 設問の指定通り年率をそのまま使用（半年換算しない）
        rf_period = rf_annual
        st.caption(
            f'シャープレシオ計算に使用する rf: {rf_period:.4f}（年率 {rf_pct:.2f}% をそのまま使用）'
        )

    # ── 学生のポートフォリオ投資比率 ─────────────
    student_weights = {}
    if selections and len(selections) >= 2:
        st.subheader('学生のポートフォリオ投資比率 (%)')
        n_sel = len(stock_names)
        default_w = round(100.0 / n_sel, 1)
        cols_w = st.columns(n_sel)
        for i, name in enumerate(stock_names):
            with cols_w[i]:
                student_weights[name] = st.number_input(
                    name, min_value=0.0, max_value=100.0,
                    value=default_w, step=0.1, format='%.1f',
                    key=f'w_{name}',
                )
        total_w = sum(student_weights.values())
        if abs(total_w - 100.0) > 1.0:
            st.warning(f'投資比率の合計: {total_w:.1f}%（合計が 100% になるよう入力してください）')
        else:
            st.caption(f'投資比率の合計: {total_w:.1f}% ✓')

    # ── 計算実行ボタン ─────────────────────────
    if not st.button('Submit and calculate'):
        return

    if len(selections) < 2:
        st.error('銘柄を 2 つ以上選択してください')
        return

    # ══════════════════════════════════════════
    st.header('課題 1')
    # ══════════════════════════════════════════

    st.subheader('課題 1.1')
    st.write('Excel シートが崩壊していなければ OK')

    # ─ 課題 1.2 ─
    st.subheader('課題 1.2')
    with st.spinner('株価データを取得中...'):
        try:
            df_price, df_ret = fetch_price_and_returns(
                tickers, stock_names, date_start, date_end
            )
        except Exception as e:
            st.error(f'データ取得エラー: {e}')
            return

    st.write('株価データ')
    st.dataframe(df_price, use_container_width=True)

    # 株価推移グラフ
    fig_p = go.Figure()
    for name in stock_names:
        fig_p.add_trace(go.Scatter(x=df_price['Date'], y=df_price[name], name=name))
    fig_p.update_layout(
        title='株価推移', height=450, width=820,
        xaxis_title='Date', yaxis_title='円',
        hovermode='x unified',
    )
    st.plotly_chart(fig_p)

    # 対数収益率グラフ
    fig_r = go.Figure()
    for name in stock_names:
        fig_r.add_trace(go.Scatter(x=df_ret['Date'], y=df_ret[name], name=name))
    fig_r.update_layout(
        title='対数収益率', height=450, width=820,
        xaxis_title='Date', yaxis_title='log-return',
        hovermode='x unified',
    )
    st.plotly_chart(fig_r)

    # ヒストグラム
    fig_h = go.Figure()
    for name in stock_names:
        fig_h.add_trace(go.Histogram(
            x=df_ret[name], name=name, opacity=0.5,
            xbins=dict(start=-0.15, end=0.15, size=0.003),
        ))
    fig_h.update_layout(
        barmode='overlay',
        title='対数収益率 ヒストグラム',
        xaxis_title='log-return', yaxis_title='度数',
        height=450, width=820,
    )
    st.plotly_chart(fig_h)

    with st.expander('元データ（対数収益率）'):
        st.dataframe(df_ret, use_container_width=True)

    # ─ 課題 1.3 ─
    st.subheader('課題 1.3')
    st.write(f'1年（{PERIOD_DAYS} 営業日）を1期間とした期待収益率・標準偏差・相関係数')

    df_nd = df_ret.drop(columns='Date')
    mu_vec    = df_nd.mean() * PERIOD_DAYS
    sigma_vec = df_nd.std()  * math.sqrt(PERIOD_DAYS)
    corr_mat  = df_nd.corr()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('**期待収益率 μ**')
        st.dataframe(mu_vec.rename('μ'))
    with col2:
        st.write('**標準偏差 σ**')
        st.dataframe(sigma_vec.rename('σ'))
    with col3:
        st.write('**相関係数行列**')
        st.dataframe(corr_mat)

    st.write('投資比率の合計が 100% になっていることを確認していれば OK')

    # ─ 課題 1.4 ─
    st.subheader('課題 1.4')

    st.write(f'**各銘柄のシャープレシオ（rf = {rf_pct:.2f}%）**')
    sr_stocks = (mu_vec - rf_period) / sigma_vec
    st.dataframe(sr_stocks.rename(f'SR (rf={rf_pct:.2f}%)'))

    st.write(f'**ポートフォリオのシャープレシオ（rf = {rf_pct:.2f}%）**')
    w_arr = np.array([student_weights[name] / 100.0 for name in stock_names])
    cov_annual = df_nd.cov().values * PERIOD_DAYS
    mu_p    = float(w_arr @ mu_vec.values)
    sigma_p = float(math.sqrt(max(w_arr @ cov_annual @ w_arr, 0.0)))
    sr_p    = (mu_p - rf_period) / sigma_p if sigma_p > 1e-10 else float('nan')

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric('投資比率', ' / '.join(f'{w*100:.1f}%' for w in w_arr))
    with col_b:
        st.metric('期待収益率 μ_p', f'{mu_p:.4f}')
    with col_c:
        st.metric('標準偏差 σ_p', f'{sigma_p:.4f}')
    with col_d:
        st.metric('シャープレシオ SR_p', f'{sr_p:.4f}')


    # ══════════════════════════════════════════
    st.header('課題 2')
    # ══════════════════════════════════════════

    st.subheader('課題 2.1')
    st.write('課題 2.2 と整合的であれば良い')

    # ─ 課題 2.2 ─
    st.subheader('課題 2.2  有効フロンティア（実際の相関係数）')

    with st.spinner(f'モンテカルロ シミュレーション（N={N}）実行中...'):
        df_mc = run_monte_carlo(df_nd, N, PERIOD_DAYS, zero_corr=False)

    fig_ef, t_sig, t_mu, t_sr = make_frontier_figure(
        df_mc, stock_names, rf_period, rf_pct,
        '有効フロンティアと資本市場線（実際の相関係数）',
    )
    st.plotly_chart(fig_ef)

    st.success(
        f'接点ポートフォリオ:  σ = {t_sig:.4f}，μ = {t_mu:.4f}，'
        f'シャープレシオ = {t_sr:.4f}'
    )
    st.write(
        '上図に学生自身のポートフォリオ（投資比率を正しく計算した点）が描画されていれば良い。'
        '接点ポートフォリオ（★）と資本市場線（赤破線）との位置関係を確認すること。'
    )

    st.subheader('課題 2.3')
    st.write('課題 2.2 と整合的であれば良い')

    st.subheader('課題 2.4')
    st.write(
        '自己のポートフォリオがポートフォリオ理論の観点から好ましいものであったか，議論ができていれば良い。'
        '接点ポートフォリオ（シャープレシオ最大）と比較して，リスク・リターンの観点から考察できているか確認する。'
    )

    # ─ 課題 2.5 ─
    st.subheader('課題 2.5  有効フロンティア（相関係数 = 0 の場合）')
    st.write('相関係数 = 0 と仮定し，課題 2.2 と同様の図を作成する。')

    with st.spinner(f'モンテカルロ シミュレーション（相関係数=0, N={N}）実行中...'):
        df_mc0 = run_monte_carlo(df_nd, N, PERIOD_DAYS, zero_corr=True)

    fig_ef0, t_sig0, t_mu0, t_sr0 = make_frontier_figure(
        df_mc0, stock_names, rf_period, rf_pct,
        '有効フロンティアと資本市場線（相関係数 = 0）',
    )
    st.plotly_chart(fig_ef0)

    st.success(
        f'接点ポートフォリオ（相関係数=0）:  σ = {t_sig0:.4f}，μ = {t_mu0:.4f}，'
        f'シャープレシオ = {t_sr0:.4f}'
    )

    delta_sigma = t_sig - t_sig0
    if delta_sigma > 0:
        st.write(
            f'✅ 接点ポートフォリオのリスク（標準偏差）が減少しました '
            f'（{t_sig:.4f} → {t_sig0:.4f}，差 = {delta_sigma:.4f}）。'
            '相関係数 = 0 のほうが分散投資の効果が大きくなるため，これは理論通りの結果です。'
        )
    else:
        st.warning(
            f'接点ポートフォリオのリスクが増加しています（{t_sig:.4f} → {t_sig0:.4f}）。'
            'MC のサンプリング誤差の可能性があります。試行回数を増やして再実行してください。'
        )


if __name__ == '__main__':
    main()
