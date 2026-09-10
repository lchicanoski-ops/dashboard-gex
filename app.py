import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Perfil GEX - B3 & EWZ", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.2rem !important; margin-bottom: 0.2rem !important; }
        h2, h3, h6 { font-size: 0.9rem !important; margin-bottom: 0.2rem !important; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    </style>
""", unsafe_allow_html=True)

def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=300)
def obter_dados_gex(ticker_symbol, taxa_juros=0.1075, multiplicador_carrego=1.0):
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Pega Preço Spot
    spot_price = None
    try:
        spot_price = ticker.fast_info['lastPrice']
    except Exception:
        pass

    if spot_price is None or np.isnan(spot_price):
        try:
            history = ticker.history(period="5d", interval="5m", prepost=True)
            if not history.empty:
                spot_price = history['Close'].iloc[-1]
        except Exception:
            pass

    if spot_price is None or np.isnan(spot_price):
        return None, None, "Não foi possível obter o preço atual do ativo."

    try:
        expirations = ticker.options
    except Exception:
        expirations = None

    if not expirations:
        return None, None, f"Sem datas de vencimento disponíveis para {ticker_symbol}."
    
    calls, puts = pd.DataFrame(), pd.DataFrame()
    
    # Varre as datas de vencimento procurando por dados válidos
    for exp_date in expirations[:4]:
        try:
            opt_chain = ticker.option_chain(exp_date)
            if not opt_chain.calls.empty and not opt_chain.puts.empty:
                c_df = opt_chain.calls[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                p_df = opt_chain.puts[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                
                # Preenche valores nulos
                c_df = c_df.fillna(0)
                p_df = p_df.fillna(0)

                # Verifica se há contratos ou volume
                if (c_df['openInterest'].sum() + p_df['openInterest'].sum() > 0) or \
                   (c_df['volume'].sum() + p_df['volume'].sum() > 0):
                    calls = c_df
                    puts = p_df
                    break
        except Exception:
            continue
            
    if calls.empty or puts.empty:
        return None, None, "Opções encontradas sem volume/posição aberta suficiente."

    T = 15 / 365.0
    gex_data = []

    # Cálculo GEX Calls
    for _, row in calls.iterrows():
        K = row['strike']
        oi = row['openInterest'] if row['openInterest'] > 0 else row['volume']
        vol = row['impliedVolatility'] if row['impliedVolatility'] > 0 else 0.20
        if oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex = gamma * oi * 100 * spot_price * 0.01
            gex_data.append({'strike': K * multiplicador_carrego, 'gex': gex})

    # Cálculo GEX Puts
    for _, row in puts.iterrows():
        K = row['strike']
        oi = row['openInterest'] if row['openInterest'] > 0 else row['volume']
        vol = row['impliedVolatility'] if row['impliedVolatility'] > 0 else 0.20
        if oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex = -(gamma * oi * 100 * spot_price * 0.01)
            gex_data.append({'strike': K * multiplicador_carrego, 'gex': gex})

    if not gex_data:
        return None, None, "Falha no cálculo do GEX."

    df_gex = pd.DataFrame(gex_data).groupby('strike')['gex'].sum().reset_index()
    
    spot_convertido = spot_price * multiplicador_carrego
    faixa_min = spot_convertido * 0.90
    faixa_max = spot_convertido * 1.10
    df_gex = df_gex[(df_gex['strike'] >= faixa_min) & (df_gex['strike'] <= faixa_max)].copy()

    if df_gex.empty:
        return None, None, "Fora da faixa operacional."

    call_wall = df_gex.loc[df_gex['gex'].idxmax()]['strike']
    put_wall = df_gex.loc[df_gex['gex'].idxmin()]['strike']
    key_level = df_gex.loc[df_gex['gex'].abs().idxmax()]['strike']

    df_gex = df_gex.sort_values('strike').reset_index(drop=True)
    df_prox = df_gex[(df_gex['strike'] >= spot_convertido * 0.93) & (df_gex['strike'] <= spot_convertido * 1.07)].copy()
    
    if not df_prox.empty:
        df_prox['sign'] = np.sign(df_prox['gex'])
        trocas_sinal = np.where(np.diff(df_prox['sign']) != 0)[0]
        
        if len(trocas_sinal) > 0:
            idx_flip = trocas_sinal[np.argmin(np.abs(df_prox.iloc[trocas_sinal]['strike'] - spot_convertido))]
            gamma_flip = df_prox.iloc[idx_flip]['strike']
        else:
            gamma_flip = df_prox.loc[df_prox['gex'].abs().idxmin()]['strike']
    else:
        gamma_flip = spot_convertido

    metricas = {
        'spot': spot_convertido,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'key_level': key_level,
        'gamma_flip': gamma_flip
    }

    return df_gex, metricas, None

def plotar_grafico_gex(df_gex, metricas, titulo, e_pontos=False):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="Gamma Exposure"
    ))

    fmt = ".0f" if e_pontos else ".2f"

    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", annotation_text=f"Spot: {metricas['spot']:{fmt}}")
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", annotation_text=f"Call Wall: {metricas['call_wall']:{fmt}}")
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", annotation_text=f"Put Wall: {metricas['put_wall']:{fmt}}")
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", annotation_text=f"Gamma Flip: {metricas['gamma_flip']:{fmt}}")

    fig.update_layout(
        title=f"{titulo} - Perfil de Gama",
        template="plotly_dark",
        xaxis_title="Gamma Exposure (GEX)",
        yaxis_title="Strike / Pontos",
        height=650,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# ------------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------------
st.title("PERFIL DE EXPOSIÇÃO DE GAMA (GEX)")

ativo = st.radio("Selecione o Mercado", ["EWZ (EUA / Brasil ETF)", "BOVA11 / WIN (B3)"], horizontal=True)

if ativo == "EWZ (EUA / Brasil ETF)":
    with st.spinner("Buscando dados de EWZ..."):
        df_gex, metricas, erro = obter_dados_gex("EWZ", taxa_juros=0.045, multiplicador_carrego=1.0)
    
    if df_gex is not None:
        col_graf, col_card = st.columns([3, 1])
        with col_graf:
            st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "EWZ"), use_container_width=True)
        with col_card:
            st.markdown("### Níveis Chave")
            st.metric("Spot Price", f"${metricas['spot']:.2f}")
            st.metric("Call Wall", f"${metricas['call_wall']:.2f}")
            st.metric("Put Wall", f"${metricas['put_wall']:.2f}")
            st.metric("Key Level", f"${metricas['key_level']:.2f}")
            st.metric("Gamma Flip", f"${metricas['gamma_flip']:.2f}")
    else:
        st.error(f"Erro ao carregar EWZ: {erro}")

else:
    with st.spinner("Buscando dados de BOVA11 e calculando projeção para o Mini Índice..."):
        taxa_di = 0.1075
        fator_win = 1000 * (1 + (taxa_di * (15 / 365)))
        df_gex, metricas, erro = obter_dados_gex("BOVA11.SA", taxa_juros=taxa_di, multiplicador_carrego=fator_win)
        
    if df_gex is not None:
        col_graf, col_card = st.columns([3, 1])
        with col_graf:
            st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "BOVA11 / Projeção WIN", e_pontos=True), use_container_width=True)
        with col_card:
            st.markdown("### Níveis em Pontos (WIN)")
            st.metric("Spot Projetado", f"{metricas['spot']:.0f} pts")
            st.metric("Call Wall", f"{metricas['call_wall']:.0f} pts")
            st.metric("Put Wall", f"{metricas['put_wall']:.0f} pts")
            st.metric("Key Level", f"{metricas['key_level']:.0f} pts")
            st.metric("Gamma Flip", f"{metricas['gamma_flip']:.0f} pts")
    else:
        st.warning(f"Indisponível no momento: {erro}")
